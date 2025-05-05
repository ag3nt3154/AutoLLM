# SQL Query to Identify Customer Inactivity Streaks

To identify customers with inactivity streaks of at least 30, 60, and 90 days, I'll use window functions to calculate the gaps between consecutive logins and then flag the longest streaks.

```sql
WITH login_sequences AS (
    SELECT 
        cif_nbr,
        datetime,
        session_id,
        LAG(datetime) OVER (PARTITION BY cif_nbr ORDER BY datetime) AS prev_login
    FROM 
        your_table_name
),

inactivity_periods AS (
    SELECT 
        cif_nbr,
        datetime,
        session_id,
        prev_login,
        DATEDIFF(day, prev_login, datetime) AS days_inactive
    FROM 
        login_sequences
    WHERE 
        prev_login IS NOT NULL
),

max_inactivity AS (
    SELECT 
        cif_nbr,
        MAX(days_inactive) AS longest_inactivity_streak
    FROM 
        inactivity_periods
    GROUP BY 
        cif_nbr
)

SELECT 
    cif_nbr,
    longest_inactivity_streak,
    CASE WHEN longest_inactivity_streak >= 30 THEN 1 ELSE 0 END AS has_30_day_streak,
    CASE WHEN longest_inactivity_streak >= 60 THEN 1 ELSE 0 END AS has_60_day_streak,
    CASE WHEN longest_inactivity_streak >= 90 THEN 1 ELSE 0 END AS has_90_day_streak
FROM 
    max_inactivity
WHERE 
    longest_inactivity_streak >= 30
ORDER BY 
    longest_inactivity_streak DESC;
```

## Alternative Version (More Detailed)

If you want to see all inactivity periods (not just the longest one) that meet your criteria:

```sql
WITH login_sequences AS (
    SELECT 
        cif_nbr,
        datetime,
        session_id,
        LAG(datetime) OVER (PARTITION BY cif_nbr ORDER BY datetime) AS prev_login
    FROM 
        your_table_name
),

inactivity_periods AS (
    SELECT 
        cif_nbr,
        datetime AS current_login,
        prev_login,
        DATEDIFF(day, prev_login, datetime) AS days_inactive,
        session_id AS current_session_id
    FROM 
        login_sequences
    WHERE 
        prev_login IS NOT NULL
)

SELECT 
    cif_nbr,
    prev_login AS last_login_before_streak,
    current_login AS first_login_after_streak,
    days_inactive,
    CASE WHEN days_inactive >= 30 THEN 1 ELSE 0 END AS meets_30_day,
    CASE WHEN days_inactive >= 60 THEN 1 ELSE 0 END AS meets_60_day,
    CASE WHEN days_inactive >= 90 THEN 1 ELSE 0 END AS meets_90_day
FROM 
    inactivity_periods
WHERE 
    days_inactive >= 30
ORDER BY 
    cif_nbr, 
    days_inactive DESC;
```

Note: The exact syntax for date difference (`DATEDIFF`) may vary depending on your database system (SQL Server, MySQL, PostgreSQL, etc.). You may need to adjust this function accordingly.

# Detailed Explanation of the SQL Query for Inactivity Streaks

Let me break down this query step by step with examples to show how it identifies customer inactivity periods.

## 1. Understanding the Data Structure

Your table has:
- `cif_nbr`: Customer identifier
- `datetime`: When the login occurred
- `session_id`: Unique identifier for each login session

Example raw data:
```
|cif_nbr| datetime           | session_id |
|-------|--------------------|------------|
| 1001  | 2023-01-01 09:00:00| sess001    |
| 1001  | 2023-01-05 10:30:00| sess002    |
| 1001  | 2023-03-01 14:00:00| sess003    |
| 1002  | 2023-01-10 11:00:00| sess004    |
| 1002  | 2023-04-15 16:00:00| sess005    |
```

## 2. The `login_sequences` CTE (Common Table Expression)

```sql
WITH login_sequences AS (
    SELECT 
        cif_nbr,
        datetime,
        session_id,
        LAG(datetime) OVER (PARTITION BY cif_nbr ORDER BY datetime) AS prev_login
    FROM 
        your_table_name
)
```

This uses the `LAG()` window function to look at each customer's previous login. The `PARTITION BY cif_nbr` ensures we only look within each customer's login history.

Example output from this CTE:
```
|cif_nbr| datetime           | session_id | prev_login         |
|-------|--------------------|------------|--------------------|
| 1001  | 2023-01-01 09:00:00| sess001    | NULL               |
| 1001  | 2023-01-05 10:30:00| sess002    | 2023-01-01 09:00:00|
| 1001  | 2023-03-01 14:00:00| sess003    | 2023-01-05 10:30:00|
| 1002  | 2023-01-10 11:00:00| sess004    | NULL               |
| 1002  | 2023-04-15 16:00:00| sess005    | 2023-01-10 11:00:00|
```

Note: The first login for each customer has NULL for `prev_login` because there was no previous login.

## 3. The `inactivity_periods` CTE

```sql
inactivity_periods AS (
    SELECT 
        cif_nbr,
        datetime AS current_login,
        prev_login,
        DATEDIFF(day, prev_login, datetime) AS days_inactive,
        session_id AS current_session_id
    FROM 
        login_sequences
    WHERE 
        prev_login IS NOT NULL
)
```

This calculates the days between consecutive logins (inactivity period) and filters out the first login (where prev_login is NULL).

Example output:
```
|cif_nbr| current_login      | prev_login         | days_inactive | current_session_id |
|-------|--------------------|--------------------|---------------|--------------------|
| 1001  | 2023-01-05 10:30:00| 2023-01-01 09:00:00| 4             | sess002            |
| 1001  | 2023-03-01 14:00:00| 2023-01-05 10:30:00| 55            | sess003            |
| 1002  | 2023-04-15 16:00:00| 2023-01-10 11:00:00| 95            | sess005            |
```

Key observations:
- Customer 1001 had a 4-day gap then a 55-day gap
- Customer 1002 had a single 95-day gap

## 4. The Final SELECT Query

```sql
SELECT 
    cif_nbr,
    prev_login AS last_login_before_streak,
    current_login AS first_login_after_streak,
    days_inactive,
    CASE WHEN days_inactive >= 30 THEN 1 ELSE 0 END AS meets_30_day,
    CASE WHEN days_inactive >= 60 THEN 1 ELSE 0 END AS meets_60_day,
    CASE WHEN days_inactive >= 90 THEN 1 ELSE 0 END AS meets_90_day
FROM 
    inactivity_periods
WHERE 
    days_inactive >= 30
ORDER BY 
    cif_nbr, 
    days_inactive DESC;
```

This produces the final result showing only inactivity periods of 30+ days, with flags for 60 and 90 day thresholds.

Final output:
```
|cif_nbr| last_login_before_streak | first_login_after_streak | days_inactive | meets_30_day | meets_60_day | meets_90_day |
|-------|--------------------------|--------------------------|---------------|--------------|--------------|--------------|
| 1001  | 2023-01-05 10:30:00      | 2023-03-01 14:00:00      | 55            | 1            | 0            | 0            |
| 1002  | 2023-01-10 11:00:00      | 2023-04-15 16:00:00      | 95            | 1            | 1            | 1            |
```

## Interpretation of Results

1. **Customer 1001**:
   - Had a 55-day inactivity period between Jan 5 and Mar 1
   - Meets the 30-day threshold but not 60 or 90 days

2. **Customer 1002**:
   - Had a 95-day inactivity period between Jan 10 and Apr 15
   - Meets all three thresholds (30, 60, and 90 days)

This query helps you identify customers who went unusually long periods without activity, which could indicate they were at risk of churning during those periods.