import datetime
import time, random

from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, MQTTOptions, DCCSOptions, EdgeData, EdgeTag, EdgeStatus, EdgeDeviceStatus, EdgeConfig, NodeConfig, DeviceConfig, AnalogTagConfig, DiscreteTagConfig, TextTagConfig
from wisepaasdatahubedgesdk.Common.Utils import RepeatedTimer

class Datahub:
  def __init__(self) -> None:
    """Init connection. Recommend to start on a new thread
    """
    self._edgeAgent = None
    edgeAgentOptions = EdgeAgentOptions(nodeId = '64a1f3ed-29ad-4efa-8f04-16668f9de7b9')
    edgeAgentOptions.connectType = constant.ConnectType['DCCS']
    dccsOptions = DCCSOptions(apiUrl = 'https://api-dccs-ensaas.education.wise-paas.com/', credentialKey = '8759510dcc9620da61f2f2ca855b85bk')
    edgeAgentOptions.DCCS = dccsOptions
    self._edgeAgent = EdgeAgent(edgeAgentOptions)
    self._edgeAgent.on_connected = self.on_connected
    self._edgeAgent.on_disconnected = self.on_disconnected
    self._edgeAgent.on_message = self.edgeAgent_on_message

    self._edgeAgent.connect()

    time.sleep(5)  # Waiting for connection to be established

  def on_connected(self, edgeAgent, isConnected):
    print("Datahub connected !")
    config = self.__generateConfig()
    self._edgeAgent.uploadConfig(action = constant.ActionType['Create'], edgeConfig = config)
    
  def on_disconnected(self, edgeAgent, isDisconnected):
    print("Datahub disconnected !")

  def edgeAgent_on_message(self, agent, messageReceivedEventArgs):
    print("Datahub edgeAgent_on_message !")

  def sendData(self, temp: float, CPU: float, RAM: float):
    """Send temperature, CPU usage and RAM to Datahub

    Args:
        temp (float): Temperature
        CPU (float): CPU usage
        RAM (float): RAM usage
    """
    edgeData = EdgeData()
    deviceId = 'lDiXKJWWZmzr'

    # Status tag
    tagName = 'status_front_gate'
    value = 1
    tag = EdgeTag(deviceId, tagName, value)
    edgeData.tagList.append(tag)

    # Temp tag
    tagName = 'temp_front_gate'
    value = temp
    tag = EdgeTag(deviceId, tagName, value)
    edgeData.tagList.append(tag)

    # CPU tag
    tagName = 'CPU_front_gate'
    value = CPU
    tag = EdgeTag(deviceId, tagName, value)
    edgeData.tagList.append(tag)

    # RAM tag
    tagName = 'RAM_front_gate'
    value = RAM
    tag = EdgeTag(deviceId, tagName, value)
    edgeData.tagList.append(tag)

    edgeData.timestamp = datetime.datetime.now()

    self._edgeAgent.sendData(edgeData)

  def __generateConfig(self):
    """Create datahub tags configuration

    Returns:
        Config: Config object to pass into EdgeAgent
    """
    config = EdgeConfig()
    deviceConfig = DeviceConfig(id = 'lDiXKJWWZmzr',
      name = 'Front_gate',
      deviceType = 'Camera',
      retentionPolicyName = '')

    discrete = DiscreteTagConfig(name = 'status_front_gate',
      readOnly = False,
      arraySize = 0,
      state0 = 'Off',
      state1 = 'On')
    deviceConfig.discreteTagList.append(discrete)

    analog1 = AnalogTagConfig(name = 'temp_front_gate',
      readOnly = False,
      arraySize = 0,
      spanHigh = 1000,
      spanLow = 0,
      engineerUnit = '',
      integerDisplayFormat = 4,
      fractionDisplayFormat = 2)
    deviceConfig.analogTagList.append(analog1)

    analog2 = AnalogTagConfig(name = 'CPU_front_gate',
      readOnly = False,
      arraySize = 0,
      spanHigh = 100000,
      spanLow = 0,
      engineerUnit = '',
      integerDisplayFormat = 4,
      fractionDisplayFormat = 2)
    deviceConfig.analogTagList.append(analog2)

    analog3 = AnalogTagConfig(name = 'RAM_front_gate',
      readOnly = False,
      arraySize = 0,
      spanHigh = 100000,
      spanLow = 0,
      engineerUnit = '',
      integerDisplayFormat = 4,
      fractionDisplayFormat = 2)
    deviceConfig.analogTagList.append(analog3)

    config.node.deviceList.append(deviceConfig)
    return config

def updateDeviceInfos(datahub):
  # datahub = Datahub()
  for i in range(1, 180):
      temp = random.random()*50
      CPU = random.random()*10
      RAM = random.random()*20
      datahub.sendData(temp, CPU, RAM)
      time.sleep(1)