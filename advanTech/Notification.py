import requests
from slowfast.utils import logging

logger = logging.get_logger(__name__)

class Notification:
    def __init__(self, username: str, password:str):
        """Init and store SSO account info

        Args:
            username (str): WISE-PaaS account username
            password (str): WISE-PaaS account password
        """
        self.user_name = username
        self.password = password

    def send_message(self, start_time: int, end_time: int, location: str, video_link: str, node_name: str, status: str):
        """Send a message via email to administrators.

        Args:
            start_time (int): Epoch time
            end_time (int): Epoch time
            location (str): Location of device
            video_link (str): Link to vide, format "https://ik.imagekit.io/vguwarriors/xxxxx.mp4"
            node_name (str): Name of that device
            status (str): On/Off
        """
        # Sign in to SSO, get cookie for authentication
        url = "https://portal-notification-hoaint-ews.education.wise-paas.com/api/v1.5/Auth"

        payload = {
            "username": self.user_name,
            "password": self.password
        }
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, json=payload, headers=headers)
        cookies = response.cookies
        # logger.info("Notification: ", str(response.text['status']))

        # Sending the email request
        url = "https://portal-notification-hoaint-ews.education.wise-paas.com/api/v1.5/Groups/send"

        payload = [
            {
                "groupId": "HUqS5Fkpsn5x",
                "subject": "A new anomaly has happened",
                "useTemplate": True,
                "variables": {
                    "location": location,
                    "start_time": start_time,
                    "end_time": end_time,
                    "link": video_link,
                    "node_name": node_name,
                    "status": status
                }
            }
        ]
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, json=payload, headers=headers, cookies=cookies)

        # logger.info("Notification:", response.text['status'])