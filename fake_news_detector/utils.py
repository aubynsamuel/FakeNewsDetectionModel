import warnings


def suppress_warnings():
    warnings.filterwarnings("ignore")


if __name__ == "__main__":
    suppress_warnings()
    print("Warnings suppressed (if run standalone).")
