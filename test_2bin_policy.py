import sys

from test_models import DQN, main


globals()["DQN"] = DQN


if __name__ == "__main__":
    if "--models" not in sys.argv:
        sys.argv.extend(["--models", "2bin"])
    main()
