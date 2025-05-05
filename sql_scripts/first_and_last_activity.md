Here's the modified SQL query that includes customers who logged in at least once between 2023-01-01 and 2024-12-31, while showing their first and last logins (which could be outside this period):

```sql
SELECT 
    t.cif_nbr,
    MIN(t.datetime) AS first_login_time,
    MAX(t.datetime) AS last_login_time
FROM 
    your_table_name t
WHERE 
    t.cif_nbr IN (
        SELECT DISTINCT cif_nbr 
        FROM your_table_name 
        WHERE datetime BETWEEN '2023-01-01' AND '2024-12-31'
    )
GROUP BY 
    t.cif_nbr
ORDER BY 
    t.cif_nbr;
```

If you want to include the session_id for the first and last logins, here's the version with window functions:

```sql
WITH filtered_customers AS (
    SELECT DISTINCT cif_nbr 
    FROM your_table_name 
    WHERE datetime BETWEEN '2023-01-01' AND '2024-12-31'
),
first_logins AS (
    SELECT 
        t.cif_nbr, 
        t.datetime AS first_login_time,
        t.session_id AS first_session_id,
        ROW_NUMBER() OVER (PARTITION BY t.cif_nbr ORDER BY t.datetime) AS rn_first
    FROM 
        your_table_name t
    WHERE 
        t.cif_nbr IN (SELECT cif_nbr FROM filtered_customers)
),
last_logins AS (
    SELECT 
        t.cif_nbr, 
        t.datetime AS last_login_time,
        t.session_id AS last_session_id,
        ROW_NUMBER() OVER (PARTITION BY t.cif_nbr ORDER BY t.datetime DESC) AS rn_last
    FROM 
        your_table_name t
    WHERE 
        t.cif_nbr IN (SELECT cif_nbr FROM filtered_customers)
)
SELECT 
    f.cif_nbr,
    f.first_login_time,
    f.first_session_id,
    l.last_login_time,
    l.last_session_id
FROM 
    first_logins f
JOIN 
    last_logins l ON f.cif_nbr = l.cif_nbr
WHERE 
    f.rn_first = 1 AND l.rn_last = 1
ORDER BY 
    f.cif_nbr;
```

This approach:
1. First identifies all customers who logged in during 2023-2024
2. Then finds their complete login history (first and last logins, regardless of date)
3. Returns the results for only those customers who logged in during the specified period