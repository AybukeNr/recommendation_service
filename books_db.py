import psycopg2

def connect_db():
    return psycopg2.connect(
        host="localhost",
        dbname="BooksDB",
        user="postgres",
        password="root",
        port=5432
    )

def get_descriptions_by_ids(book_ids):
    conn = connect_db()
    cursor = conn.cursor()
    placeholders = ','.join(['%s'] * len(book_ids))
    query = f"SELECT id, description FROM book WHERE id IN ({placeholders})"
    cursor.execute(query, tuple(book_ids))
    results = cursor.fetchall()
    conn.close()
    return results  
