# SQL Query for Daily Customer Activity

Here's a SQL query that will count the number of distinct sessions per customer per day:

```sql
SELECT 
    cif_nbr,
    CAST(datetime AS DATE) AS date,
    COUNT(DISTINCT session_id) AS num_sessions
FROM 
    your_table_name
GROUP BY 
    cif_nbr,
    CAST(datetime AS DATE)
ORDER BY 
    cif_nbr,
    CAST(datetime AS DATE);
```

## Explanation:

1. `CAST(datetime AS DATE)` - This converts the datetime to just the date portion (removing the time component)
2. `COUNT(DISTINCT session_id)` - Counts unique session IDs for each customer on each day
3. `GROUP BY cif_nbr, CAST(datetime AS DATE)` - Groups the results by customer and date
4. `ORDER BY` - Sorts the results by customer and then by date

## Example:

If your input data looks like:
```
|cif_nbr| datetime           | session_id |
|-------|--------------------|------------|
| 1001  | 2023-01-01 09:00:00| sess001    |
| 1001  | 2023-01-01 10:30:00| sess002    |
| 1001  | 2023-01-01 14:00:00| sess001    | (same session later in day)
| 1001  | 2023-01-02 11:00:00| sess003    |
| 1002  | 2023-01-01 16:00:00| sess004    |
```

The output would be:
```
|cif_nbr| date       | num_sessions |
|-------|------------|-------------|
| 1001  | 2023-01-01 | 2           | (sess001 and sess002)
| 1001  | 2023-01-02 | 1           | (sess003)
| 1002  | 2023-01-01 | 1           | (sess004)
```

Note: The exact syntax for date casting (`CAST(datetime AS DATE)`) might vary slightly depending on your database system:
- MySQL: `CAST(datetime AS DATE)` or `DATE(datetime)`
- SQL Server: `CAST(datetime AS DATE)`
- PostgreSQL: `datetime::date`
- Oracle: `TRUNC(datetime)`