import json
import paho.mqtt.client as mqtt
import mysql.connector
from config import MQTT_USERNAME, MQTT_PASSWORD, MQTT_HOST, MQTT_PORT


class DBManager:
    def __init__(self, database='drawbo_v2', host="101.201.75.83", user="root", password="askroot"):
        self.connection = mysql.connector.connect(
            user=user, 
            password=password,
            host=host, # name of the mysql service as set in the docker compose file
            database=database,
            auth_plugin='mysql_native_password'
        )
        self.cursor = self.connection.cursor()

    
    def query_devices(self):
        self.cursor.execute('SELECT * FROM device')
        rec = []
        for c in self.cursor:
            rec.append(c)
        return rec


    def select_topic(self, deviceid):
        self.cursor.execute("SELECT topic FROM device WHERE deviceid = %s ;", (deviceid,))

        return self.cursor.fetchone()[0]

    
    def count_device(self):
        self.cursor.execute('SELECT count(1) FROM device')
        return self.cursor.fetchone()[0]




def send_message_to_topic(topic: str, push_dict: dict)->int:
    """
    :param topic:
    :param push_dict: 结构
    :return: errcode
    """
    push_body = str.encode(json.dumps(push_dict))
    # 初始化MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.username_pw_set(username=MQTT_USERNAME, password=MQTT_PASSWORD)
    client.connect(MQTT_HOST, MQTT_PORT)

    #计划异步实现 xiaojuzi 20231023 qos=0 最多一次  1 最少一次 2 只有一次
    
    client.publish(topic, push_body, 1)

    return 0


if __name__ == "__main__":
    conn = DBManager()
    num = conn.count_device()
    print(num)

    topic = conn.select_topic('8c000c6d60064842652')
    print(topic)

    devices = conn.query_devices()
    print(len(devices))
