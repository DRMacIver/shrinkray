CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    status TEXT CHECK (status IN ('pending', 'completed', 'cancelled')),
    amount NUMERIC(10, 2) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

INSERT INTO customers (id, name, email) VALUES
    (1, 'Alice', 'alice@example.com'),
    (2, 'Bob', 'bob@example.com');

INSERT INTO orders (id, customer_id, order_date, status, amount) VALUES
    (1, 1, '2023-01-15', 'completed', 42.50),
    (2, 1, '2023-02-03', 'pending', 19.99),
    (3, 2, '2023-02-20', 'completed', 8.00);
