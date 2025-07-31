from datetime import datetime
from calendar import monthrange

def get_first_and_last_days(start_date_str, end_date_str):
    """
    Calculates the first and last day of each month between two dates.

    Args:
        start_date_str (str): The start date in YYYY-MM-DD format.
        end_date_str (str): The end date in YYYY-MM-DD format.

    Returns:
        list: A list of tuples, where each tuple contains the first and last
              day of the month as strings in YYYY-MM-DD format.
              Returns an empty list if the input dates are invalid or if the
              start date is after the end date.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        return []  # Return empty list for invalid date format

    if start_date > end_date:
        return []  # Return empty list if start date is after end date

    result = []
    year = start_date.year
    month = start_date.month

    while True:
        # Calculate the first day of the current month
        first_day = datetime(year, month, 1)

        # Calculate the last day of the current month
        _, last_day_num = monthrange(year, month)  # Get the number of days in the month
        last_day = datetime(year, month, last_day_num)

        if first_day > end_date:
            break # Exit the loop if the first day is after the end date.

        # Add the first and last day to the result
        result.append((first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")))

        # Move to the next month
        month += 1
        if month > 12:
            month = 1
            year += 1

    return result

if __name__ == "__main__":
    # Example usage
    start_date = "2023-10-25"
    end_date = "2024-03-10"
    result = get_first_and_last_days(start_date, end_date)
    print(f"First and last days of each month between {start_date} and {end_date}:")
    for first_day, last_day in result:
        print(f"First Day: {first_day}, Last Day: {last_day}")

    print("\nTesting with invalid date format:")
    invalid_result = get_first_and_last_days("2023-10-25", "2024/03/10") # Invalid format
    print(f"Result with invalid date format: {invalid_result}") # Should print []

    print("\nTesting with start date after end date:")
    invalid_result_2 = get_first_and_last_days("2024-03-10", "2023-10-25")
    print(f"Result with start date after end date: {invalid_result_2}")  # Should print []
