"""
This is a module used for SQL database interaction
func: create_database - creates predefined database
func: insert_data(table, **data) - inserts data into corresponding table
func: post_entry_exist(reddit_id) - checks if a post with specified reddit_id exists in the 'post' table -> True/False
func: update_entry(reddit_id, **data) - updates data in a table with a specified reddit_id in viable table
func: find_pot_post - returns tuple with a chosen post entry
func: get_table_data - return columns and data of a table
func: list_media - returns list of all suitable media
"""

import sqlite3
import os

def create_database():
    """
    Creates a predefined database for the YouTubeBOT
    """
    # connection
    conn = sqlite3.connect("YT_data.db")
    cursor = conn.cursor()

    # stock_video table
    cursor.execute("""
    CREATE TABLE stock_video (
    stock_video_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE CHECK (LENGTH(name) > 0),
    author VARCHAR(50),
    attribution VARCHAR(200),
    url VARCHAR(200),
    theme VARCHAR(20) NOT NULL CHECK (LENGTH(theme) > 0)
    );""")

    # stock_music table
    cursor.execute("""
    CREATE TABLE stock_music (
    stock_music_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE CHECK (LENGTH(name) > 0),
    author VARCHAR(50),
    attribution VARCHAR(200),
    url VARCHAR(200),
    theme VARCHAR(20) NOT NULL CHECK (LENGTH(theme) > 0)
    );""")

    # post table
    cursor.execute("""
    CREATE TABLE post (
    reddit_id VARCHAR(10) NOT NULL PRIMARY KEY CHECK (LENGTH(reddit_id) > 0),
    title VARCHAR (200) NOT NULL CHECK (LENGTH(title) > 0),
    sub VARCHAR(50) NOT NULL CHECK (LENGTH(sub) > 0),
    author VARCHAR(50) NOT NULL CHECK (LENGTH(author) > 0),
    length INTEGER NOT NULL,
    url VARCHAR(200) NOT NULL CHECK (LENGTH(url) > 0),
    score INTEGER NOT NULL,
    num_comments INTEGER NOT NULL,
    ratio FLOAT NOT NULL,
    NSFW BOOLEAN NOT NULL
    );""")

    # video table
    cursor.execute("""
    CREATE TABLE video (
    video_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reddit_id VARCHAR(10),
    title VARCHAR (200) NOT NULL,
    length INTEGER NOT NULL,
    channel VARCHAR(50) NOT NULL,
    FOREIGN KEY(reddit_id) REFERENCES post(reddit_id)
    );""")

    # video_stock_video table
    cursor.execute("""
    CREATE TABLE video_stock_video (
    video_id INTEGER,
    stock_video_id INTEGER,
    PRIMARY KEY (video_id, stock_video_id),
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (stock_video_id) REFERENCES stock_video(stock_video_id)
    );""")

    # video_stock_music table
    cursor.execute("""
    CREATE TABLE video_stock_music (
    video_id INTEGER,
    stock_music_id INTEGER,
    PRIMARY KEY (video_id, stock_music_id),
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (stock_music_id) REFERENCES stock_music(stock_music_id)
    );""")

    conn.commit()

def insert_data(table, used_videos = [], used_music = [], **data):
    """
    Inserts data into a specified table
    param: table - table to insert into
    param: **data - data to insert; key = column, value = value
    opt param: used_videos - only when inserting a video
    opt param: used_music - only when inserting a video
    """
    # Connection
    conn = sqlite3.connect("YT_data.db")
    cursor = conn.cursor()

    # Create query
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data.values()))
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

    # Execute the query with the data values
    try:
        cursor.execute(query, tuple(data.values()))

        # When inserting video ('used_' params != None) -> update junction tables (inserted new video)
        if table == 'video':
            # Find video_id of the last inserted video
            video_id = cursor.lastrowid

            # Updating video_stock_video table
            for stock_video_id in used_videos:
                # Insert new entry into the junction table
                cursor.execute("INSERT INTO video_stock_video (video_id, stock_video_id) VALUES (?, ?)", (video_id, stock_video_id))

            # Updating video_stock_music table
            for stock_music_id in used_music:
                # Insert new entry into the junction table
                cursor.execute("INSERT INTO video_stock_music (video_id, stock_music_id) VALUES (?, ?)", (video_id, stock_music_id))

    finally:
        conn.commit()
        conn.close()

def post_entry_exists(reddit_id):
    """
    Checks if a post is already in the database
    param: reddit_id - reddit_id of the checked post
    out: True/False
    """
    # Connection
    conn = sqlite3.connect("YT_data.db")
    cursor = conn.cursor()

    # Create query
    query = "SELECT * FROM post WHERE reddit_id = ?"
    cursor.execute(query,(reddit_id,))
    result = cursor.fetchone()

    conn.close()

    if result is None:
        return False
    else:
        return True

def update_entry(table, reddit_id, **data):
    """
    Updates value(s) in a column(s) of a post or video
    param: table - any table that uses reddit_id = post, video
    param: reddit_id - reddit_id of the updated post
    param: **data - data to update; key = column, value = value
    """
    # Connection
    conn = sqlite3.connect("YT_data.db")
    cursor = conn.cursor()
    updates = []

    query = f"""
    UPDATE {table} SET
    """

    for key in data.keys():
        updates.append(f"\t{key} = ?")

    query += ",\n".join(updates)
    query += f"\nWHERE reddit_id = '{reddit_id}'\n"

    cursor.execute(query, tuple(data.values()))
    conn.commit()
    conn.close()

def find_top_post(minimal_post_length, subs):
    """
    Finds the post with best score based on minimal length
    out: reddit_id, title, subreddit and author of the post
    Returns only posts where NSFW = False and post was not used to create a video already!
    """
    # Connection
    conn = sqlite3.connect("YT_data.db")
    cursor = conn.cursor()

    # Query
    query = """
    SELECT p.reddit_id, p.title, p.sub, p.author
    FROM post p
    LEFT JOIN video v ON p.reddit_id = v.reddit_id
    WHERE v.reddit_id IS NULL AND p.NSFW = False AND p.length >= ?
    """

    # Adding subreddit filter if provided
    query += f" AND p.sub IN ({','.join('?' for _ in subs)})"

    query += " ORDER BY score DESC;"

    # Execute query
    cursor.execute(query, (minimal_post_length, *subs))

    # Fetch query result
    top_post = cursor.fetchone()

    # Close connection and return top post
    conn.close()
    return top_post

def get_table_data(table):
    """
    Returns data from a specified table
    param: table - table to get data from
    out: data - list of tuples, each tuplle is a database entry
    out: columns - list of all column names
    """
    conn = sqlite3.connect('YT_data.db')
    cursor = conn.cursor()

    # Get data
    cursor.execute(f"SELECT * FROM {table}")
    data = cursor.fetchall()

    # Get column names
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]

    conn.close()
    return data, columns

def list_media(media:str, theme:str):
    """
    Lists media of the specified type
    param: media - table name of the media -> stock_video, stock_music
    param: theme - theme listed in the database
    out: [(id, name, attribution), ...] - list of tuples with ids, names and attributions of the found media
    """
    conn = sqlite3.connect('YT_data.db')
    cursor = conn.cursor()

    query = f"""
    SELECT {media}_id, name, attribution
    FROM {media}
    WHERE theme = ?
    """
    query_data = (theme, )
    cursor.execute(query, query_data)

    result = cursor.fetchall()

    return result

# When the module is imported try to create the database and dump all post text files - they are missing from the database
try:
    create_database()
    for post in os.listdir('Posts'):
        os.remove(os.path.join('Posts', post))
except:
    pass

if __name__ == '__main__':
    # Insert a showcase post into the database and confirm by printing
    # insert_data('post', reddit_id = '!FAKE_ID', score = 31384561837, title = 'Lorem ipsum dolor sit amet? Consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua!', ratio = 0.5, num_comments = 5, sub = "AITAH", author = "Me", length = 31, url = "fakeurl.com", NSFW = False)
    print(find_top_post(9000, ['AITAH']))

    # For deleting entries in tables:
    conn = sqlite3.connect("YT_data.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM video;")
    cursor.execute("DELETE FROM video_stock_music;")
    cursor.execute("DELETE FROM video_stock_video;")
    #cursor.execute("DELETE FROM post;")
    #cursor.execute("DELETE FROM stock_video;")

    conn.commit()
    conn.close()
    pass
