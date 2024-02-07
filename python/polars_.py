import polars as pl


df = pl.DataFrame({"field_1": ['a', 'b', 'c', 'd', 'd']})

def val(params):
    print(f"params={params}")

    return 1
vals = df.select(pl.col("field_1").unique())

df = df.lazy().with_columns(
    field_1_num= val(vals)
).collect()

print(df)