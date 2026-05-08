
def calculate_yield(input_weight, output_weight):
    if input_weight == 0:
        return 0
    return (output_weight / input_weight) * 100


def main():
    yield_percent = calculate_yield(100, 92)
    print(f"Process yield: {yield_percent}%")

if __name__ == "__main__":
    main()
