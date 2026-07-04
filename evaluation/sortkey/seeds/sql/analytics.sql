WITH monthly_sales AS (
    SELECT
        customer_id,
        strftime('%Y-%m', order_date) AS month,
        SUM(amount) AS total
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id, month
),
ranked AS (
    SELECT
        customer_id,
        month,
        total,
        ROW_NUMBER() OVER (PARTITION BY month ORDER BY total DESC) AS rank
    FROM monthly_sales
)
SELECT
    r.month,
    c.name,
    r.total
FROM ranked AS r
JOIN customers AS c ON c.id = r.customer_id
WHERE r.rank <= 3
ORDER BY r.month, r.total DESC;
