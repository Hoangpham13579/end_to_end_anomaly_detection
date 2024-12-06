# importing psycopg2 module
import psycopg2

class Database:
    def __init__(self, database: str, user: str, password: str, host: str, port: str = '5432'):
        """Init and store database config

        Args:
            database (str): Database name
            user (str): Database master user, default "postgres"
            password (str): Database password
            host (str): Database host IP
            port (str, optional): Database port. Defaults to '5432'.
        """
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
    
    def add_row(self, start_time: int, end_time: int, location: str, video_link: str, device_status: str, node_name: str):
        """Add a row into existing table in database

        Args:
            start_time (int): Epoch time
            end_time (int): Epoch time
            location (str): Location of device
            video_link (str): Link to vide, format "https://ik.imagekit.io/vguwarriors/xxxxx.mp4"
            node_name (str): Name of that device
            status (str): On/Off
        """
        device_status = True if "On" or "on" or "ON" or "oN" else False

        # establishing the connection
        conn = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

        # creating a cursor object
        cursor = conn.cursor()

        print('CHECK!!! ', start_time)
        cursor.execute(f"INSERT INTO anomaly (start_time, end_time, location, video_link, node_status, node_name) values ({start_time}, {end_time}, '{location}', '{video_link}', {device_status}, '{node_name}');")


        print("Database: List has been inserted to table successfully...")

        # Commit your changes in the database
        conn.commit()

        # Closing the connection
        conn.close()