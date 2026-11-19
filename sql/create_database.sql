-- Run while connected to the default postgres database as a PostgreSQL admin.
-- CREATE DATABASE cannot run inside a transaction block, so execute separately if needed.
CREATE DATABASE ds01_churn;
