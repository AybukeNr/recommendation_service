import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "BooksDB",
    "user": "postgres",
    "password": "root"
}

# Eşleşen kitapların açıklamalarını dbden al
def get_recommended_books():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = "SELECT id, description FROM book WHERE description IS NOT NULL"
    cursor.execute(query)
    books = cursor.fetchall()

    cursor.close()
    conn.close()
    return books

# Ziyaret edilen kitapların açıklamalarını dbden al
def get_visited_books_by_ids(book_ids):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    sql = "SELECT id, description FROM books WHERE id = ANY(%s) AND description IS NOT NULL"
    cursor.execute(sql, (book_ids,))
    books = cursor.fetchall()

    cursor.close()
    conn.close()
    return books
