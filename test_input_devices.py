from livekit.rtc import MediaDevices
import pprint

devices = MediaDevices()

print("--- Available Input Devices ---")
for device in devices.list_input_devices():
    pprint.pprint(device)  # inspect raw object
