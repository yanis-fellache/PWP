import sqlite3

db = sqlite3.connect('Insomnia_Authentication.db')
cursor = db.cursor()


def set_up():
    cursor.execute('''CREATE TABLE IF NOT EXISTS Log_In (
    id integer primary key, Username varchar(255), Password varchar(255))''')
    return cursor


def log_in(username, password):
    data = cursor.execute('''SELECT * FROM Log_In''')
    for row in data:
        if username == row[1]:
            if password == row[2]:
                return 'Logged In'
            return 'Wrong Password'
    return 'User Does Not Exist'


def create_user(username, password):
    cursor.execute(f"INSERT INTO Log_In (id, username, password) VALUES (NULL, '{username}', '{password}')")


db.commit()
db.close()
