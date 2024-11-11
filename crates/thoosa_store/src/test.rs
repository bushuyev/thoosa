use rusqlite::{Connection, Result};

struct SqlStore {
    conn: Connection,
}

impl SqlStore {
    fn from_path(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        Ok(SqlStore { conn })
    }

    fn in_mem() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Ok(SqlStore { conn })
    }

    fn execute(&self, query: &str) -> Result<()> {
        self.conn.execute(query, [])?;
        Ok(())
    }
}

#[test]
fn test_attach() -> Result<()> {
    // Step 1: Create file-backed and in-memory databases
    std::fs::remove_file("/tmp/test_db.db3").unwrap_or(());
    std::fs::remove_file("/tmp/test_db_tmp.db3").unwrap_or(());
    let store_file = SqlStore::from_path("/tmp/test_db.db3")?;
    let store_tmp = SqlStore::from_path("/tmp/test_db_tmp.db3")?;

    // Step 2: Create a table in both databases
    store_file.execute("CREATE TABLE IF NOT EXISTS data (id INTEGER PRIMARY KEY, value TEXT)")?;
    store_tmp.execute("CREATE TABLE IF NOT EXISTS data (id INTEGER PRIMARY KEY, value TEXT)")?;

    // Step 3: Insert some values into the table in the memory database
    let insert_query = "INSERT INTO data (value) VALUES (?1)";
    let values = ["Value 1", "Value 2", "Value 3"];
    for value in &values {
        store_file.conn.execute(insert_query, &[value])?;
    }

    // Step 4: Attach the memory database to the file-backed database
    store_file
        .conn
        .execute("ATTACH DATABASE '/tmp/test_db_tmp.db3' AS tmp", [])?;



    // Step 5: Move data from the memory database table to the file database table
    store_file.execute(
        "INSERT INTO data (value) SELECT value FROM tmp.data",
    )?;

    // Step 6: Detach the memory database
    store_file.conn.execute("DETACH DATABASE tmp", [])?;

    // Verify the transfer
    let mut stmt = store_file.conn.prepare("SELECT value FROM data")?;
    let rows: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;

    assert_eq!(rows, values);

    Ok(())
}
