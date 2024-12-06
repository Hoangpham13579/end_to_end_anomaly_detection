from azure.storage.blob import BlobServiceClient, ContentSettings
import os
import datetime, threading


class AzureStorage:

    def start_upload_thread(self, container_name:str, file_path:str):
        """Start a thread to upload video to Azure Blob Storage

        Args:
            container_name (str): Name of the blob container. Each location should have its own container, e.g. binhduong, hcm, hanoi,...
            file_path (str): Path to local video. Ending with .mp4
        """
        upAzure_thread = threading.Thread(
            target=self.upload_blob, args=(container_name, file_path), daemon=False
            )
        upAzure_thread.start()

    def upload_blob(self, container_name:str, file_path:str):
        """DO NOT USE UNLESS YOU INTENTIONALLY WANT TO USE THIS
        This function will upload a video to Azure blob storage. But it will block the main thread. It was designed to be handled by "start_upload_thread" so do not use it unless you intentionally want to.

        Args:
            container_name (str): Name of the blob container. Each location should have its own container, e.g. binhduong, hcm, hanoi,...
            file_path (str): Path to local video. Ending with .mp4
        """
        try:
            connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not connect_str:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)

            # Create a container
            container_name = container_name
            local_file_name = file_path.split('/')[-1]
            upload_file_path = file_path
            try:
                blob_service_client.create_container(container_name)
            except:
                pass

            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

            print(f' AzureStorage: {datetime.datetime.now()} Uploading to Azure Storage as blob: {local_file_name}')

            # Upload the created file
            with open(upload_file_path, "rb") as data:
                content_settings = ContentSettings(content_type='video/mp4')
                blob_client.upload_blob(data, content_settings=content_settings)#, connection_timeout=14400)
            print(f' AzureStorage: {datetime.datetime.now()} Finished uploading')

        except Exception as ex:
            print(f' AzureStorage: Exception:')
            print(f' AzureStorage: {ex}')

# a = AzureStorage()
# a.start_upload_thread("binhduong", "data/Arson012_x264.mp4")