import argparse
import json
import random
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from util.sumo_setup import configure_sumo

configure_sumo()
import sumo_rl

from agent.Fixed_agent import Fiexed
from env.wrap.random_block import BlockStreet
import traci


REFERENCE_METRICS = [
    "TTT",
    "avg_queue",
    "N_des",
    "N_in",
    "N_out",
    "latent_demand",
    "residual_demand",
    "service_ratio",
    "throughput",
    "avg_time_loss",
    "wall_clock_sec",
    "sim_step_count",
]

TWO_BIN_DIAGNOSTIC_METRICS = [
    "KH_mean",
    "KH_min",
    "KH_max",
    "KV_mean",
    "KV_min",
    "KV_max",
    "horizontal_green_total_seconds",
    "vertical_green_total_seconds",
    "cycle_count",
]

MACLIGHT_DENSITY_OFFSET = 9
MACLIGHT_HORIZONTAL_DENSITY_INDICES = (3, 4, 5, 9, 10, 11)
MACLIGHT_VERTICAL_DENSITY_INDICES = (0, 1, 2, 6, 7, 8)


class DQN(torch.nn.Module):
    def __init__(self, StateDim=2, ActionDim=11, l1=32, l2=64, state_dim=None, action_dim=None):
        super().__init__()
        state_dim = StateDim if state_dim is None else state_dim
        action_dim = ActionDim if action_dim is None else action_dim
        self.fc1 = torch.nn.Linear(in_features=state_dim, out_features=l1)
        self.fc2 = torch.nn.Linear(in_features=l1, out_features=l2)
        self.out = torch.nn.Linear(in_features=l2, out_features=action_dim)

    def forward(self, obs):
        x = obs.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.out(x)


class TorchPolicyWrapper:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, obs):
        obs_tensor = torch.as_tensor(
            obs,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        q_values = self.model(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())


class EpisodeMetricTracker:
    def __init__(self, desired_demand_total=None, step_seconds=5):
        self.step_seconds = int(step_seconds)
        self.desired_demand_total = desired_demand_total
        self.start_wall_clock = time.perf_counter()
        self.total_travel_time = 0.0
        self.queue_sum = 0.0
        self.queue_count = 0
        self.n_in = 0
        self.n_out = 0
        self.departed_seen = set()
        self.arrived_seen = set()
        self.active_seen = set()
        self.total_time_loss_network_sum = 0.0
        self.sim_step_count = 0
        self.wall_clock_sec = 0.0

    def update_after_env_step(self, total_queue):
        traci_conn = getattr(self, "traci_conn", traci)
        veh_ids = list(traci_conn.vehicle.getIDList())
        self.total_travel_time += float(len(veh_ids) * self.step_seconds)
        self.queue_sum += float(total_queue)
        self.queue_count += 1

        for veh_id in veh_ids:
            if veh_id not in self.active_seen:
                self.active_seen.add(veh_id)
                if veh_id not in self.departed_seen:
                    self.departed_seen.add(veh_id)
                    self.n_in += 1

        for veh_id in list(traci_conn.simulation.getDepartedIDList()):
            if veh_id not in self.departed_seen:
                self.departed_seen.add(veh_id)
                self.n_in += 1

        for veh_id in list(traci_conn.simulation.getArrivedIDList()):
            if veh_id not in self.arrived_seen:
                self.arrived_seen.add(veh_id)
                self.n_out += 1

        network_time_loss = 0.0
        for veh_id in veh_ids:
            try:
                network_time_loss += float(traci_conn.vehicle.getTimeLoss(veh_id))
            except Exception:
                pass
        self.total_time_loss_network_sum += network_time_loss * self.step_seconds
        self.sim_step_count += self.step_seconds

    def finalize(self):
        self.wall_clock_sec = time.perf_counter() - self.start_wall_clock

    def as_dict(self):
        latent = None
        service_ratio = None
        if self.desired_demand_total is not None:
            latent = max(int(self.desired_demand_total) - int(self.n_in), 0)
            service_ratio = 0.0 if self.desired_demand_total <= 0 else float(self.n_out) / float(self.desired_demand_total)
        residual = max(int(self.n_in) - int(self.n_out), 0)

        return {
            "TTT": float(self.total_travel_time),
            "avg_queue": float(self.queue_sum / max(self.queue_count, 1)),
            "N_des": None if self.desired_demand_total is None else int(self.desired_demand_total),
            "N_in": int(self.n_in),
            "N_out": int(self.n_out),
            "latent_demand": None if latent is None else int(latent),
            "residual_demand": int(residual),
            "service_ratio": None if service_ratio is None else float(service_ratio),
            "throughput": int(self.n_out),
            "avg_time_loss": float(self.total_time_loss_network_sum / max(self.sim_step_count, 1)),
            "wall_clock_sec": float(self.wall_clock_sec),
            "sim_step_count": int(self.sim_step_count),
        }


def make_env(args):
    env = sumo_rl.parallel_env(
        net_file="env/map/ff.net.xml",
        route_file=f"env/map/ff_{args.level}.rou.xml",
        num_seconds=args.seconds,
        use_gui=False,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )
    if args.task == "block":
        env = BlockStreet(env, args.block_num, args.seconds)
    return env


def make_sumo_env(args):
    return sumo_rl.SumoEnvironment(
        net_file="env/map/ff.net.xml",
        route_file=f"env/map/ff_{args.level}.rou.xml",
        num_seconds=args.seconds,
        use_gui=False,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )


def get_agent_names(env):
    return env.agent_name if hasattr(env, "agent_name") else env.possible_agents


def get_action_space(env, agent):
    base_env = env.env if hasattr(env, "env") else env
    return base_env.action_space(agent)


def load_2bin_policy(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"2bin policy not found: {model_path}")

    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        return TorchPolicyWrapper(loaded, device)

    if isinstance(loaded, dict):
        model = DQN(StateDim=2, ActionDim=11, l1=32, l2=64)
        model.load_state_dict(loaded)
        return TorchPolicyWrapper(model, device)

    raise ValueError(f"Unsupported 2bin policy format: {type(loaded)}")


def maclight_obs_to_2bin(obs, density_mapping=None):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[0] >= 33:
        lane_densities = obs[MACLIGHT_DENSITY_OFFSET:MACLIGHT_DENSITY_OFFSET + 12]
        h_indices = density_mapping["H_indices"] if density_mapping else MACLIGHT_HORIZONTAL_DENSITY_INDICES
        v_indices = density_mapping["V_indices"] if density_mapping else MACLIGHT_VERTICAL_DENSITY_INDICES
        phase1_load = float(np.mean(lane_densities[list(h_indices)]) * 150.0)
        phase2_load = float(np.mean(lane_densities[list(v_indices)]) * 150.0)
        return np.array([phase1_load, phase2_load], dtype=np.float32)
    elif obs.shape[0] >= 12:
        lane_densities = obs[:12]
    else:
        lane_densities = obs

    half = max(len(lane_densities) // 2, 1)
    phase1_load = float(np.mean(lane_densities[:half]) * 150.0)
    phase2_load = float(np.mean(lane_densities[half:]) * 150.0)
    return np.array([phase1_load, phase2_load], dtype=np.float32)


def route_file_for(args):
    return Path(f"env/map/ff_{args.level}.rou.xml")


def count_desired_demand(route_file, seconds):
    if not route_file.exists():
        return None
    count = 0
    for _event, elem in ET.iterparse(route_file, events=("end",)):
        if elem.tag == "vehicle":
            depart = elem.attrib.get("depart")
            if depart is None:
                count += 1
            else:
                try:
                    if float(depart) < float(seconds):
                        count += 1
                except ValueError:
                    count += 1
            elem.clear()
    return count


def classify_incoming_lane_axis(env, tls_id, lane_id):
    # Fall back to lane shape geometry because SUMO TraCI does not expose edge
    # endpoint IDs. For this grid, the first/last lane-shape coordinates are
    # enough to identify approach orientation.
    shape = env.sumo.lane.getShape(lane_id)
    if len(shape) < 2:
        return "H"
    x0, y0 = shape[0]
    x1, y1 = shape[-1]
    return "H" if abs(x1 - x0) > abs(y1 - y0) else "V"


def lane_record(env, tls_id, density_index, lane_id):
    shape = env.sumo.lane.getShape(lane_id)
    x0, y0 = shape[0] if shape else (None, None)
    x1, y1 = shape[-1] if shape else (None, None)
    return {
        "density_index": density_index,
        "lane_id": lane_id,
        "incoming_edge": lane_id.rsplit("_", 1)[0],
        "shape_start": None if x0 is None else [float(x0), float(y0)],
        "shape_end": None if x1 is None else [float(x1), float(y1)],
        "axis": classify_incoming_lane_axis(env, tls_id, lane_id),
    }


def build_density_axis_mapping(env, tls_id):
    lanes = env.traffic_signals[tls_id].lanes
    records = [lane_record(env, tls_id, idx, lane_id) for idx, lane_id in enumerate(lanes)]
    h_indices = tuple(record["density_index"] for record in records if record["axis"] == "H")
    v_indices = tuple(record["density_index"] for record in records if record["axis"] == "V")
    if len(h_indices) + len(v_indices) != len(lanes):
        raise ValueError(f"{tls_id}: not all density lanes classified as H/V")
    return {
        "records": records,
        "H_indices": h_indices,
        "V_indices": v_indices,
        "H_lanes": [record["lane_id"] for record in records if record["axis"] == "H"],
        "V_lanes": [record["lane_id"] for record in records if record["axis"] == "V"],
    }


def build_two_phase_states(env, tls_id):
    links = env.sumo.trafficlight.getControlledLinks(tls_id)
    horizontal = []
    vertical = []
    link_records = []
    for idx, link in enumerate(links):
        if not link:
            horizontal.append(False)
            vertical.append(False)
            link_records.append({
                "link_index": idx,
                "incoming_lane": None,
                "outgoing_lane": None,
                "axis": None,
                "empty": True,
            })
            continue
        incoming_lane = link[0][0]
        outgoing_lane = link[0][1]
        axis = classify_incoming_lane_axis(env, tls_id, incoming_lane)
        horizontal.append(axis == "H")
        vertical.append(axis == "V")
        if horizontal[-1] == vertical[-1]:
            shape = env.sumo.lane.getShape(incoming_lane)
            raise ValueError(
                f"{tls_id}: controlled link {idx} is not exclusively H/V: "
                f"{incoming_lane}->{outgoing_lane}, shape={shape}"
            )
        link_records.append({
            "link_index": idx,
            "incoming_lane": incoming_lane,
            "outgoing_lane": outgoing_lane,
            "axis": axis,
            "empty": False,
        })

    horizontal_state = "".join("G" if is_h else "r" for is_h in horizontal)
    vertical_state = "".join("G" if is_v else "r" for is_v in vertical)
    if len(horizontal_state) != len(links) or len(vertical_state) != len(links):
        raise ValueError(f"{tls_id}: phase state length does not match controlled link count")
    if set(horizontal_state + vertical_state) - {"G", "r"}:
        raise ValueError(f"{tls_id}: two-phase states contain non green/red chars")
    for idx, record in enumerate(link_records):
        if record["empty"]:
            continue
        if (horizontal_state[idx] == "G") == (vertical_state[idx] == "G"):
            raise ValueError(f"{tls_id}: non-empty link {idx} is not complementary across H/V phases")
    return {
        "horizontal_state": horizontal_state,
        "vertical_state": vertical_state,
        "link_records": link_records,
        "H_links": [record for record in link_records if record["axis"] == "H"],
        "V_links": [record for record in link_records if record["axis"] == "V"],
    }


def install_two_phase_programs(env, two_phase_states):
    for tls_id, phase_info in two_phase_states.items():
        horizontal_state = phase_info["horizontal_state"]
        vertical_state = phase_info["vertical_state"]
        logic = env.sumo.trafficlight.getAllProgramLogics(tls_id)[0]
        logic.type = 0
        logic.phases = [
            env.sumo.trafficlight.Phase(10, horizontal_state),
            env.sumo.trafficlight.Phase(10, vertical_state),
        ]
        env.sumo.trafficlight.setProgramLogic(tls_id, logic)
        env.sumo.trafficlight.setRedYellowGreenState(tls_id, horizontal_state)


def apply_blocking_for_raw_sumo(env, block_state, current_time):
    if block_state is None:
        return

    blockable_edges = block_state["blockable_edges"]
    if current_time % 200 != 0:
        for edge_id in block_state["rd_id"]:
            env.sumo.edge.setMaxSpeed(blockable_edges[edge_id], 0.5)
    else:
        for edge_id in block_state["rd_id"]:
            env.sumo.edge.setMaxSpeed(blockable_edges[edge_id], 13.89)
        block_state["rd_id"] = torch.randperm(len(blockable_edges))[:block_state["block_num"]].tolist()

    for vehicle_id in env.sumo.vehicle.getIDList():
        env.sumo.vehicle.rerouteTraveltime(vehicle_id)


def make_block_state(args):
    if args.task != "block":
        return None
    blockable_edges = [
        "B2C2", "B3C3", "C1C2", "C2B2", "C2C1", "C2C3",
        "C2D2", "C3B3", "C3C2", "C3C4", "C3D3", "C4C3",
        "D1D2", "D2C2", "D2D1", "D2D3", "D2E2", "D3C3",
        "D3D2", "D3D4", "D3E3", "D4D3", "E2D2", "E3D3",
    ]
    return {
        "blockable_edges": blockable_edges,
        "block_num": args.block_num,
        "rd_id": torch.randperm(len(blockable_edges))[:args.block_num].tolist(),
    }


def print_two_bin_diagnostics(density_mappings, two_phase_states):
    print("\n[2bin mapping diagnostics]")
    print("H/V density grouping is built from SUMO-RL traffic_signals[ts].lanes order.")
    print("TLS two-phase grouping is built from SUMO controlled links using incoming lane geometry.")
    for tls_id in sorted(density_mappings):
        density = density_mappings[tls_id]
        phase = two_phase_states[tls_id]
        density_h_set = set(density["H_lanes"])
        density_v_set = set(density["V_lanes"])
        link_h_set = {record["incoming_lane"] for record in phase["H_links"]}
        link_v_set = {record["incoming_lane"] for record in phase["V_links"]}
        h_consistent = density_h_set == link_h_set
        v_consistent = density_v_set == link_v_set

        print(f"\nTLS {tls_id}")
        print("  density lanes:")
        for record in density["records"]:
            print(
                f"    idx={record['density_index']:02d} lane={record['lane_id']} "
                f"edge={record['incoming_edge']} axis={record['axis']} "
                f"shape={record['shape_start']}->{record['shape_end']}"
            )
        print(f"  horizontal density lanes: {density['H_lanes']}")
        print(f"  vertical density lanes  : {density['V_lanes']}")
        print(
            "  horizontal controlled links: "
            f"{[(r['link_index'], r['incoming_lane'], r['outgoing_lane']) for r in phase['H_links']]}"
        )
        print(
            "  vertical controlled links  : "
            f"{[(r['link_index'], r['incoming_lane'], r['outgoing_lane']) for r in phase['V_links']]}"
        )
        print(f"  horizontal_state: {phase['horizontal_state']}")
        print(f"  vertical_state  : {phase['vertical_state']}")
        print(f"  density/link H consistent: {h_consistent}")
        print(f"  density/link V consistent: {v_consistent}")
        if not h_consistent or not v_consistent:
            raise ValueError(f"{tls_id}: density lane grouping and controlled-link grouping are inconsistent")

    print("\n[2bin checks]")
    print("  H/V density grouping matches TLS phase grouping for all traffic lights.")
    print("  Two-phase states contain only 'G' and 'r'; no yellow states are installed.")
    print("  2bin action is interpreted as horizontal green duration in seconds, not as a MacLight phase id.")
    print("  2bin path bypasses env.step(action) and uses raw setRedYellowGreenState()+simulationStep().")
    print("  Lane order is verified at runtime from SUMO-RL traffic_signals[ts].lanes.")


def reset_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(env, agent_name, action_fn, seed):
    state, done, truncated = env.reset(seed=seed)[0], False, False
    episode_return = 0.0
    episode_reward_sum = 0.0
    episode_reward_mean_sum = 0.0
    simulation_time = 0
    info = None
    action_hist = {}
    tracker = EpisodeMetricTracker(
        desired_demand_total=getattr(env, "desired_demand_total", None),
        step_seconds=getattr(env, "metric_step_seconds", 5),
    )

    while not (done | truncated):
        action = action_fn(state, simulation_time)
        for selected_action in action.values():
            action_key = int(selected_action)
            action_hist[action_key] = action_hist.get(action_key, 0) + 1
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        simulation_time += 5
        reward_values = list(reward.values())
        mean_reward = float(np.mean(reward_values))
        episode_return += mean_reward
        episode_reward_sum += float(np.sum(reward_values))
        episode_reward_mean_sum += mean_reward
        tracker.update_after_env_step(info[agent_name[0]]["system_total_stopped"])
        done = all(list(done.values()))
        truncated = all(list(truncated.values()))

    tracker.finalize()
    reference_metrics = tracker.as_dict()
    return {
        "Return": episode_return,
        "waiting_list": info[agent_name[0]]["system_total_waiting_time"],
        "queue_list": info[agent_name[0]]["system_total_stopped"],
        "speed_list": info[agent_name[0]]["system_mean_speed"],
        **reference_metrics,
        "episode_reward_sum": float(episode_reward_sum),
        "episode_reward_mean_per_step": float(episode_reward_mean_sum / max(tracker.queue_count, 1)),
        "action_hist_json": json.dumps(action_hist, ensure_ascii=False),
    }


def test_idqn(args, ckpt_path, train_seed):
    reset_random(args.eval_seed_start)
    env = make_env(args)
    env.desired_demand_total = count_desired_demand(route_file_for(args), args.seconds)
    env.metric_step_seconds = args.metric_step_seconds
    agent_name = get_agent_names(env)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    agents = checkpoint["agent"]

    for agent in agents.values():
        agent.epsilon = 0.0
        agent.device = args.device
        agent.q_net.to(args.device)
        agent.q_net.eval()

    def action_fn(state, _simulation_time):
        with torch.no_grad():
            return {agt: agents[agt].take_action(state[agt]) for agt in agent_name}

    rows = []
    for episode in range(args.eval_episodes):
        eval_seed = args.eval_seed_start + episode
        metrics = run_episode(env, agent_name, action_fn, eval_seed)
        rows.append({
            "Algorithm": "IDQN",
            "Train seed": train_seed,
            "Eval episode": episode + 1,
            "Eval seed": eval_seed,
            **metrics,
            "Checkpoint": str(ckpt_path),
        })
        print(f"IDQN train_seed={train_seed} eval_episode={episode + 1}: "
              f"return={metrics['Return']:.2f}, waiting={metrics['waiting_list']:.0f}, "
              f"queue={metrics['queue_list']}, speed={metrics['speed_list']:.4f}, "
              f"TTT={metrics['TTT']:.0f}, avg_queue={metrics['avg_queue']:.2f}, "
              f"throughput={metrics['throughput']}")

    env.close()
    return rows


def test_ippo(args, ckpt_path, train_seed):
    reset_random(args.eval_seed_start)
    env = make_env(args)
    env.desired_demand_total = count_desired_demand(route_file_for(args), args.seconds)
    env.metric_step_seconds = args.metric_step_seconds
    agent_name = get_agent_names(env)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    agents = checkpoint["agent"]

    for agent in agents.values():
        agent.device = args.device
        agent.actor.to(args.device)
        agent.actor.eval()
        if hasattr(agent, "critic"):
            agent.critic.to(args.device)
            agent.critic.eval()

    def action_fn(state, _simulation_time):
        actions = {}
        with torch.no_grad():
            for agt in agent_name:
                state_tensor = torch.as_tensor(state[agt], dtype=torch.float32, device=args.device)
                probs = agents[agt].actor(state_tensor)
                actions[agt] = int(torch.argmax(probs).item())
        return actions

    rows = []
    for episode in range(args.eval_episodes):
        eval_seed = args.eval_seed_start + episode
        metrics = run_episode(env, agent_name, action_fn, eval_seed)
        rows.append({
            "Algorithm": "IPPO",
            "Train seed": train_seed,
            "Eval episode": episode + 1,
            "Eval seed": eval_seed,
            **metrics,
            "Checkpoint": str(ckpt_path),
        })
        print(f"IPPO train_seed={train_seed} eval_episode={episode + 1}: "
              f"return={metrics['Return']:.2f}, waiting={metrics['waiting_list']:.0f}, "
              f"queue={metrics['queue_list']}, speed={metrics['speed_list']:.4f}, "
              f"TTT={metrics['TTT']:.0f}, avg_queue={metrics['avg_queue']:.2f}, "
              f"throughput={metrics['throughput']}")

    env.close()
    return rows


def test_fixed(args):
    reset_random(args.eval_seed_start)
    env = make_env(args)
    env.desired_demand_total = count_desired_demand(route_file_for(args), args.seconds)
    env.metric_step_seconds = args.metric_step_seconds
    agent_name = get_agent_names(env)
    action_dict = {agent: 0 for agent in agent_name}
    action_space = {agent: list(range(get_action_space(env, agent).n)) for agent in agent_name}
    agent = Fiexed(args.fixed_flag, action_space, action_dict, agent_name)

    def action_fn(_state, simulation_time):
        return agent.take_action(simulation_time)

    rows = []
    for episode in range(args.eval_episodes):
        eval_seed = args.eval_seed_start + episode
        metrics = run_episode(env, agent_name, action_fn, eval_seed)
        rows.append({
            "Algorithm": "Fixed",
            "Train seed": None,
            "Eval episode": episode + 1,
            "Eval seed": eval_seed,
            **metrics,
            "Checkpoint": None,
        })
        print(f"Fixed eval_episode={episode + 1}: "
              f"return={metrics['Return']:.2f}, waiting={metrics['waiting_list']:.0f}, "
              f"queue={metrics['queue_list']}, speed={metrics['speed_list']:.4f}, "
              f"TTT={metrics['TTT']:.0f}, avg_queue={metrics['avg_queue']:.2f}, "
              f"throughput={metrics['throughput']}")

    env.close()
    return rows


def current_observations(env):
    return {ts: env.traffic_signals[ts].compute_observation() for ts in env.ts_ids}


def run_2bin_episode(env, agent_name, policy, args, seed, desired_demand_total, print_diagnostics=False):
    env.reset(seed=seed)
    density_mappings = {ts: build_density_axis_mapping(env, ts) for ts in agent_name}
    two_phase_states = {ts: build_two_phase_states(env, ts) for ts in agent_name}
    if print_diagnostics:
        print_two_bin_diagnostics(density_mappings, two_phase_states)
    install_two_phase_programs(env, two_phase_states)
    block_state = make_block_state(args)
    state = current_observations(env)

    tracker = EpisodeMetricTracker(desired_demand_total=desired_demand_total, step_seconds=1)
    tracker.traci_conn = env.sumo
    action_hist = {a: 0 for a in range(11)}
    episode_return = 0.0
    episode_reward_sum = 0.0
    episode_reward_mean_sum = 0.0
    cycle_count = 0
    kh_values = []
    kv_values = []
    raw_action_hist = {a: 0 for a in range(11)}
    cycle_log = []
    observation_probe = []
    horizontal_green_total_seconds = 0
    vertical_green_total_seconds = 0
    previous_queue = float(env._get_system_info()["system_total_stopped"])

    current_time = 0
    while current_time < args.seconds:
        two_bin_actions = {}
        for agt in agent_name:
            two_bin_obs = maclight_obs_to_2bin(state[agt], density_mappings[agt])
            kh_values.append(float(two_bin_obs[0]))
            kv_values.append(float(two_bin_obs[1]))
            action = int(policy.predict(two_bin_obs))
            action = max(0, min(10, action))
            two_bin_actions[agt] = action
            action_hist[action] += 1
            raw_action_hist[action] += 1

        cycle_seconds = min(10, args.seconds - current_time)
        horizontal_seconds = {agt: min(two_bin_actions[agt], cycle_seconds) for agt in agent_name}
        vertical_seconds = {agt: cycle_seconds - horizontal_seconds[agt] for agt in agent_name}
        horizontal_green_total_seconds += int(sum(horizontal_seconds.values()))
        vertical_green_total_seconds += int(sum(vertical_seconds.values()))
        if len(cycle_log) < 20:
            cycle_log.append({
                "cycle": cycle_count,
                "actions": dict(two_bin_actions),
                "horizontal_seconds": dict(horizontal_seconds),
                "vertical_seconds": dict(vertical_seconds),
            })
        for cycle_second in range(cycle_seconds):
            apply_blocking_for_raw_sumo(env, block_state, current_time)
            for agt in agent_name:
                horizontal_state = two_phase_states[agt]["horizontal_state"]
                vertical_state = two_phase_states[agt]["vertical_state"]
                signal_state = horizontal_state if cycle_second < two_bin_actions[agt] else vertical_state
                env.sumo.trafficlight.setRedYellowGreenState(agt, signal_state)
            env.sumo.simulationStep()
            current_time += 1
            info = env._get_system_info()
            tracker.update_after_env_step(info["system_total_stopped"])
            if print_diagnostics and len(observation_probe) < 20:
                probe_state = current_observations(env)
                probe_agent = agent_name[0]
                probe_obs = maclight_obs_to_2bin(probe_state[probe_agent], density_mappings[probe_agent])
                observation_probe.append({
                    "sim_second": current_time,
                    "agent": probe_agent,
                    "KH": float(probe_obs[0]),
                    "KV": float(probe_obs[1]),
                })

        state = current_observations(env)
        current_queue = float(env._get_system_info()["system_total_stopped"])
        reward = -(current_queue - previous_queue)
        previous_queue = current_queue
        mean_reward = reward / max(len(agent_name), 1)
        episode_return += mean_reward
        episode_reward_sum += reward
        episode_reward_mean_sum += mean_reward
        cycle_count += 1

    tracker.finalize()
    info = env._get_system_info()
    kh_arr = np.asarray(kh_values, dtype=float)
    kv_arr = np.asarray(kv_values, dtype=float)
    if print_diagnostics:
        print("\n[2bin observation update probe]")
        print(json.dumps(observation_probe, ensure_ascii=False))
        print("[2bin cycle scheduler probe]")
        print(json.dumps(cycle_log, ensure_ascii=False))
        print("[2bin final semantic checks]")
        print("  No yellow: confirmed by generated states containing only 'G' and 'r'.")
        print("  Action semantics: a seconds H green, 10-a seconds V green; no phase-id mapping used.")
        print("  Raw observation update: KH/KV probe above is read after raw simulationStep().")
    if kh_arr.size and (np.mean(kh_arr) < 1e-6 or np.mean(kh_arr) > 149.0):
        print(f"[warning] 2bin KH_mean={np.mean(kh_arr):.4f}; check state scaling/lane grouping.")
    if kv_arr.size and (np.mean(kv_arr) < 1e-6 or np.mean(kv_arr) > 149.0):
        print(f"[warning] 2bin KV_mean={np.mean(kv_arr):.4f}; check state scaling/lane grouping.")
    return {
        "Return": float(episode_return),
        "waiting_list": info["system_total_waiting_time"],
        "queue_list": info["system_total_stopped"],
        "speed_list": info["system_mean_speed"],
        **tracker.as_dict(),
        "KH_mean": float(np.mean(kh_arr)) if kh_arr.size else None,
        "KH_min": float(np.min(kh_arr)) if kh_arr.size else None,
        "KH_max": float(np.max(kh_arr)) if kh_arr.size else None,
        "KV_mean": float(np.mean(kv_arr)) if kv_arr.size else None,
        "KV_min": float(np.min(kv_arr)) if kv_arr.size else None,
        "KV_max": float(np.max(kv_arr)) if kv_arr.size else None,
        "episode_reward_sum": float(episode_reward_sum),
        "episode_reward_mean_per_step": float(episode_reward_mean_sum / max(cycle_count, 1)),
        "action_hist_json": json.dumps(action_hist, ensure_ascii=False),
        "raw_2bin_action_hist_json": json.dumps(raw_action_hist, ensure_ascii=False),
        "horizontal_green_total_seconds": int(horizontal_green_total_seconds),
        "vertical_green_total_seconds": int(vertical_green_total_seconds),
        "cycle_count": int(cycle_count),
        "cycle_log_json": json.dumps(cycle_log, ensure_ascii=False),
    }


def test_2bin(args):
    reset_random(args.eval_seed_start)
    env = make_sumo_env(args)
    desired_demand_total = count_desired_demand(route_file_for(args), args.seconds)
    agent_name = env.ts_ids
    policy = load_2bin_policy(Path(args.two_bin_model), args.device)

    rows = []
    for episode in range(args.eval_episodes):
        eval_seed = args.eval_seed_start + episode
        reset_random(eval_seed)
        metrics = run_2bin_episode(
            env,
            agent_name,
            policy,
            args,
            eval_seed,
            desired_demand_total,
            print_diagnostics=(episode == 0),
        )
        rows.append({
            "Algorithm": "2bin",
            "Train seed": None,
            "Eval episode": episode + 1,
            "Eval seed": eval_seed,
            **metrics,
            "Checkpoint": str(Path(args.two_bin_model)),
        })
        print(f"2bin eval_episode={episode + 1}: "
              f"return={metrics['Return']:.2f}, waiting={metrics['waiting_list']:.0f}, "
              f"queue={metrics['queue_list']}, speed={metrics['speed_list']:.4f}, "
              f"TTT={metrics['TTT']:.0f}, avg_queue={metrics['avg_queue']:.2f}, "
              f"throughput={metrics['throughput']}")

    env.close()
    return rows


def write_results(args, rows):
    scene = f"{args.task}_{args.level}"
    out_dir = Path(args.out_dir) / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(rows)
    detail_path = out_dir / "test_detail.csv"
    summary_path = out_dir / "test_summary.csv"

    data.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_metrics = [
        "Return",
        "waiting_list",
        "queue_list",
        "speed_list",
        *REFERENCE_METRICS,
        *TWO_BIN_DIAGNOSTIC_METRICS,
        "episode_reward_sum",
        "episode_reward_mean_per_step",
    ]
    summary_metrics = [metric for metric in summary_metrics if metric in data.columns]
    summary = data.groupby("Algorithm")[summary_metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("waiting_list_mean")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policies and Fixed Time.")
    parser.add_argument("--models", nargs="+", default=["IDQN", "Fixed"], choices=["IPPO", "IDQN", "Fixed", "2bin"])
    parser.add_argument("-t", "--task", default="block", choices=["regular", "block"])
    parser.add_argument("-l", "--level", default="normal", choices=["normal", "hard"])
    parser.add_argument("-b", "--block_num", default=8, type=int)
    parser.add_argument("--seconds", default=3600, type=int)
    parser.add_argument("--eval-episodes", default=5, type=int)
    parser.add_argument("--eval-seed-start", default=100, type=int)
    parser.add_argument("--ippo-train-seeds", nargs="+", default=[42, 43], type=int)
    parser.add_argument("--idqn-train-seeds", nargs="+", default=[42, 43], type=int)
    parser.add_argument("--ckpt-root", default="ckpt")
    parser.add_argument("--out-dir", default="data/test_data")
    parser.add_argument("--two-bin-model", default="ckpt/2bin_policy.pt")
    parser.add_argument("--fixed-flag", default=40, type=int)
    parser.add_argument("--metric-step-seconds", default=5, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scene = f"{args.task}_{args.level}"
    system_type = sys.platform
    rows = []

    if "IPPO" in args.models:
        for train_seed in args.ippo_train_seeds:
            ckpt_path = Path(args.ckpt_root) / scene / "IPPO" / f"{train_seed}_IPPO_{system_type}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"IPPO checkpoint not found: {ckpt_path}")
            rows.extend(test_ippo(args, ckpt_path, train_seed))

    if "IDQN" in args.models:
        for train_seed in args.idqn_train_seeds:
            ckpt_path = Path(args.ckpt_root) / scene / "IDQN" / f"{train_seed}_IDQN_{system_type}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"IDQN checkpoint not found: {ckpt_path}")
            rows.extend(test_idqn(args, ckpt_path, train_seed))

    if "Fixed" in args.models:
        rows.extend(test_fixed(args))

    if "2bin" in args.models:
        rows.extend(test_2bin(args))

    write_results(args, rows)


if __name__ == "__main__":
    main()
