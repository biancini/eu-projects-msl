"""Tests with duck DB."""

import duckdb

filename = "/Users/andrea.biancinigmail.com/Library/CloudStorage/OneDrive-Personale/Documenti/Documenti personali/Mio Patrimonio.xlsx"

conn = duckdb.connect()
conn.execute("INSTALL 'excel'; LOAD 'excel';")

result = conn.execute(f"""
    SELECT * FROM read_xlsx(
        '{filename}',
        sheet='Movimenti',
        header=TRUE,
        range='3:1000000',
        all_varchar=TRUE,
        stop_at_empty=TRUE)
""").fetch_df()

print(result)
