<template>
  <div>
    <el-container>
      <el-main style="background-color: #e9eef3;border-radius: 1.5vw;">
        <div style="display: flex;">
          <div style="width: 70%;">
            <div style="height: 100%;width: 98%;">
              <div
                  style="text-align: center;font-size: larger;font-weight: 900;color: black;background-color: #e9eef3; height: 5vh;line-height: 5vh;">摄像头画面
              </div>
              <div v-if="isShowImg1 && isLogin !== 2">
                <img :src="imgurl1" alt="electron-vue" style="width:100%;height:90%">
                <div v-if="isDroneCameraOpen"
                     style="position: absolute; bottom: 34px; left: -20px; width: 35%; height: 35%;">
                  <img :src="imgurl2" alt="drone-camera" style="width:100%; height:100%; object-fit: contain;">
                </div>
              </div>
              <div v-if="!isShowImg1 && isLogin !== 2">
                <img style="height: 100%;width: 100%;object-fit: contain;" src="~@/assets/bg.png" alt="electron-vue">
              </div>
              <div v-if="isShowImg2 && isLogin === 2">
                <img :src="imgurl2" alt="electron-vue" style="width:100%;height:90%">
              </div>
              <div v-if="!isShowImg2 && isLogin === 2">
                <img style="height: 100%;width: 100%;object-fit: contain;" src="~@/assets/bg.png" alt="electron-vue">
              </div>
            </div>


          </div>
          <div style="width: 30%;height: 90vh;">
            <div style="height: 4vh;"></div>
            <div style="height: 86vh;background-color: #ffffff;border-radius: 1.5vw;">

              <!---------------------系统登陆------------------------>
              <div v-if="isLogin === 0">
                <div style="text-align: center;font-weight: 900;line-height: 10vh;font-size: 25px;">系统登录</div>

                <div><img src="~@/assets/lg.png" alt="electron-vue" style="width: 80%;margin-left: 10%;"></div>
                <div style="height: 10vh;"></div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-input style="width: 80%;" v-model="name" placeholder="请输入名字（限英文字符）"></el-input>
                </div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 80%;font-weight: 600;" @click="storageFace">注册人脸
                  </el-button>
                </div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 80%;font-weight: 600;" @click="login">登录</el-button>
                </div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera1">
                    {{ isPcCameraOpen ? '关闭' : '打开' }}电脑摄像头
                  </el-button>
                </div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 80%;font-weight: 600;" @click="redirectToAdmin">人脸管理
                  </el-button>
                </div>
              </div>


              <!---------------------地面站面板------------------------>
              <div v-if="isLogin === 1">
                <div style="text-align: center;font-weight: 900;line-height: 10vh;font-size: 25px;">地面站面板</div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-select v-model="value1" placeholder="模式选择" @change="change">
                    <el-option
                        v-for="item in options1"
                        :key="item.value"
                        :label="item.label"
                        :value="item.value">
                    </el-option>
                  </el-select>
                </div>

                <!---------------------人脸------------------------>
                <div v-if="value1===1">
                  <div style="height: 20vh;"></div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-input style="width: 80%;" v-model="name" placeholder="请输入名字（限英文字符）"></el-input>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="storageFace">录入人脸</el-button>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera1">
                      {{ isPcCameraOpen ? '关闭' : '打开' }}电脑摄像头
                    </el-button>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="redirectToAdmin">人脸管理</el-button>
                  </div>
                </div>

                <!---------------------手势------------------------>
                <div v-if="value1===2">
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                       @click="openHand">
                    <div style="width: 40%;text-align: center;">手势识别</div>
                    <el-switch
                        v-model="isHand"
                        active-text="开启"
                        inactive-text="关闭">
                    </el-switch>
                  </div>

                  <!-- 手势命令照片展示 -->
                  <div
                      style="margin-top: 2vh; display: flex; flex-wrap: wrap; justify-content: space-evenly; gap: 1vw;">
                    <img src="~@/assets/gesture_control.png" alt="gesture_label"
                         style="height: 70%; width: 70%; border-radius: 1vw;"/>
                  </div>
                </div>

                <!----------------------无人机状态----------------------------->
                <div v-if="value1===3">

                 <div
                    v-for="(item, index) in stateItems"
                    :key="index"
                    class="status-item"
                  >
                    <div class="label">{{ item.label }}：</div>
                    <div class="value">
                      {{
                        currentState[item.key] !== null && currentState[item.key] !== undefined
                          ? item.format(currentState[item.key])
                          : "N/A"
                      }}
                    </div>
                  </div>

                  <div style="height: 8vh;"></div>

                  <div style="font-weight: 900;margin-top: -3vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="toggleDroneConnection"
                               v-loading.fullscreen.lock="fullscreenLoading">
                      {{ isDroneConnected ? '断开无人机链接' : '连接无人机' }}
                    </el-button>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera2">
                      {{ isDroneCameraOpen ? '关闭' : '打开' }}无人机摄像头
                    </el-button>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="switchboard">切换至无人机
                    </el-button>
                  </div>
                </div>
              </div>


              <!---------------------无人机面板--------------------------->
              <div v-if="isLogin === 2"><!---->
                <div style="text-align: center;font-weight: 900;line-height: 10vh;font-size: 25px;">无人机面板</div>
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-select v-model="value2" placeholder="模式选择" @change="change">
                    <el-option
                        v-for="item in options2"
                        :key="item.value"
                        :label="item.label"
                        :value="item.value">
                    </el-option>
                  </el-select>
                </div>

                <!------------------无人机状态----------------------------->
                <div v-if="value2===1">

                  <div
                    v-for="(item, index) in stateItems"
                    :key="index"
                    class="status-item"
                  >
                    <div class="label">{{ item.label }}：</div>
                    <div class="value">
                      {{
                        currentState[item.key] !== null && currentState[item.key] !== undefined
                          ? item.format(currentState[item.key])
                          : "N/A"
                      }}
                    </div>
                  </div>

                  <div style="height: 8vh;"></div>

                  <div style="font-weight: 900;margin-top: -3vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="toggleDroneConnection"
                               v-loading.fullscreen.lock="fullscreenLoading">
                      {{ isDroneConnected ? '断开无人机连接' : '连接无人机' }}
                    </el-button>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera2">
                      {{ isDroneCameraOpen ? '关闭' : '打开' }}无人机摄像头
                    </el-button>
                  </div>
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="switchboard">切换至地面站
                    </el-button>
                  </div>
                </div>

                <!---------------------人脸跟随------------------------->
                <div v-if="value2===2">

                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-switch
                      v-model="isVisualizationEnabled"
                      active-text="开启可视化"
                      inactive-text="关闭可视化"
                      @change="toggleVisualization"
                    >
                    </el-switch>
                  </div>

                  <div style="height: 40vh; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div style="font-weight: 900; margin-bottom: 2vh; text-align: center;">
                      使用提示：
                    </div>
                    <ul style="list-style-type: none; padding: 0; margin: 0; text-align: left; width: 80%;">
                      <li style="margin-bottom: 1vh;">1、请确保无人机有足够的飞行空间</li>
                      <li style="margin-bottom: 1vh;">2、请勿在人多处使用，尽量确保无人机视角内只有一人</li>
                      <li style="margin-bottom: 1vh;">3、最佳跟踪距离为1m左右</li>
                    </ul>
                  </div>

                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="faceTrack">
                      {{ isFaceTracking ? '停止' : '开启' }}人脸跟随
                    </el-button>
                  </div>
                </div>

                <!--------------------按键控制---------------------------->
                <div v-if="value2===3">

                  <div v-for="(group, index) in buttonGroups" :key="index" class="button-group">
                    <el-button
                      v-for="button in group"
                      :key="button.command"
                      type="primary"
                      class="custom-button"
                      @click="sendCommand(button.command)"
                    >
                      {{ button.label }}
                    </el-button>
                  </div>

                  <div
                      style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;line-height: 7vh;font-size: 2.5vh;">
                    <div style="width: 35%;text-align: center;">移动速度</div>
                    <el-input-number v-model="move_speed" :min="30" :max="100" label="描述文字"></el-input-number>
                  </div>

                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;" @click="updateSpeed">设置</el-button>
                  </div>
                </div>

                <!-------------------语音控制-------------------------->
                <div v-if="value2===4">
                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                       @click="openVoice">
                    <div style="width: 40%;text-align: center;">使能语音控制</div>
                    <el-switch
                        v-model="isVoice"
                        active-text="开启"
                        inactive-text="关闭">
                    </el-switch>
                  </div>

                  <div
                      style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;font-size: 4vh;">
                    <div style="width: 40%;text-align: center;">当前指令</div>
                    {{ getVoiceAction }}
                  </div>

                  <div style="margin-top: 2vh; text-align: center;">
                    <div style="font-weight: 900; margin-bottom: 1vh;">可用语音指令：</div>
                    <div style="display: grid; grid-template-columns: repeat(2, auto); gap: 8px; width: 80%; margin: 0 auto;">
                      <el-tag
                        v-for="(cmd, index) in commandsMap"
                        :key="index"
                        type="info"
                        style="text-align: center;"
                      >
                        {{ cmd }}
                      </el-tag>
                    </div>
                  </div>

                  <div style="height: 3vh;"></div>

                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: center;">
                    <el-progress style="width: 80%;" :percentage="progress" :status="progressStatus" :text-inside="true" :stroke-width="26"/>
                  </div>

                  <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                    <el-button type="primary" style="width: 80%;font-weight: 600;"
                               @click="recordVoice"
                    :disabled="!isVoice || isRecordingVoice"
                    :title="
                          !isVoice
                            ? '请先开启语音控制'
                            : isRecordingVoice
                              ? '正在录音，请稍后'
                              : '开始录音'"
                    >
                    {{ isRecordingVoice ? '录音中...' : '开始录音' }}
                    </el-button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>


<script>
import SystemInformation from './LandingPage/SystemInformation'

export default {
  name: 'landing-page',
  components: {SystemInformation},
  data() {
    return {
      fullscreenLoading: false,
      isLogin: 0,
      name: '',
      // isLogin: 2,

      imgurl1_: 'http://localhost:8000/pc_video',
      isShowImg1: false,
      isPcCameraOpen: false,

      imgurl2_: 'http://localhost:8000/drone_video',
      isShowImg2: false,
      isDroneCameraOpen: false,
      isDroneConnected: false,

      isFaceTracking: false,
      isFace: false,
      isPoint: false,
      isAlign: false,

      isHand: false,
      isHandPoint: false,

      commandsMap: {
        "takeoff": "起飞",
        "land": "降落",
        "up": "上升",
        "down": "下降",
        "forward": "前进",
        "backward": "后退",
        "left": "向左飞",
        "right": "向右飞",
        "rotate_left": "向左转",
        "rotate_right": "向右转",
      },

      isVoice: false,
      voiceAction: "",
      isRecordingVoice: false,
      progress: 0,
      progressStatus: 'success',

      timer: null,
      CurrentStateTimer: null,

      timestamp: Date.now(),

      move_speed: 50,

      stateItems: [
        { label: "电池电量", key: "bat", format: (val) => `${val}%` },
        { label: "飞行时间", key: "time", format: (val) => `${val}s` },
        { label: "气压计高度", key: "baro", format: (val) => `${val}cm` },
        { label: "下表面高度", key: "tof", format: (val) => `${val}cm` },
        {
          label: "速度(X,Y,Z)",
          key: "vg",
          format: (val) => `(${val.vgx}, ${val.vgy}, ${val.vgz})`
        },
        {
          label: "姿态(P,R,Y)",
          key: "attitude",
          format: (val) => `(${val.pitch}, ${val.roll}, ${val.yaw})`
        },
        { label: "信号强度", key: "wifi", format: (val) => `${val}` },
      ],

      buttonGroups: [
        [{ command: 'takeoff', label: '起飞' }, { command: 'land', label: '降落' }],
        [{ command: 'up', label: '上升' }, { command: 'down', label: '下降' }],
        [{ command: 'forward', label: '前进' }, { command: 'backward', label: '后退' }],
        [{ command: 'left', label: '左移' }, { command: 'right', label: '右移' }],
        [{ command: 'rotate_left', label: '左旋' }, { command: 'rotate_right', label: '右旋' }]
      ],

      currentState: {
        templ: 0,
        temph: 0,
        tof: 0,
        h: 0,
        bat: 0,
        baro: 0.0,
        time: 0,
        attitude: { pitch: 0, roll: 0, yaw: 0 },
        ag: { agx: 0.0, agy: 0.0, agz: 0.0 },
        vg: { vgx: 0.0, vgy: 0.0, vgz: 0.0 },
        wifi: "0"
      },

      options1: [{
        value: 1,
        label: '人脸录入管理'
      }, {
        value: 2,
        label: '手势控制'
      },
        {
          value: 3,
          label: '无人机状态'
        }],

      options2: [
        {
          value: 1,
          label: '无人机状态'
        },
        {
          value: 2,
          label: '人脸跟随'
        },
        {
          value: 3,
          label: '按键控制'
        },
        {
          value: 4,
          label: '语音控制'
        }],
      value1: 1,
      value2: 1,

      isVisualizationEnabled: false
    }
  },

  mounted() {
    //
  },

  computed: {
    //
    imgurl1() {
      return `${this.imgurl1_}?t=${this.timestamp}`;
    },
    imgurl2() {
      return `${this.imgurl2_}?t=${this.timestamp}`;
    },

    getVoiceAction() {
      if (this.voiceAction in this.commandsMap) {
        this.sendCommand(this.voiceAction );
        return this.commandsMap[this.voiceAction]; // 返回中文指令
      } else {
        return "";
      }
    }
  },

  watch: {
    //
  },
  methods: {

    switchboard() {
      if (this.isLogin === 1) {
        this.isLogin = 2
      } else if (this.isLogin === 2) {
        this.isLogin = 1
      }
      console.log(this.isLogin)
    },

    getCurrentState() {
      if (!this.isDroneConnected) {
        // 如果无人机未连接，清除定时器并返回
        clearInterval(this.CurrentStateTimer);
        return;
      }

      this.CurrentStateTimer = setInterval(() => {
        this.$http.get('http://127.0.0.1:8000/get_current_state').then(response => {
          if (response.data.status === 1) {
            this.currentState.attitude.pitch = response.data.tello_state.pitch;
            this.currentState.attitude.roll = response.data.tello_state.roll;
            this.currentState.attitude.yaw = response.data.tello_state.yaw;

            this.currentState.vg.vgx = response.data.tello_state.vgx;
            this.currentState.vg.vgy = response.data.tello_state.vgy;
            this.currentState.vg.vgz = response.data.tello_state.vgz;

            this.currentState.bat = response.data.tello_state.bat;
            this.currentState.time = response.data.tello_state.time;

            this.currentState.templ = response.data.tello_state.templ;
            this.currentState.temph = response.data.tello_state.temph;

            this.currentState.h = response.data.tello_state.h;
            this.currentState.tof = response.data.tello_state.tof;
            this.currentState.baro = response.data.tello_state.baro;

            this.currentState.ag.agx = response.data.tello_state.agx;
            this.currentState.ag.agy = response.data.tello_state.agy;
            this.currentState.ag.agz = response.data.tello_state.agz;

            this.currentState.wifi = response.data.tello_state.wifi;
          } else {
            this.$message({
              message: '获取状态参数失败',
              type: 'error'
            });
            clearInterval(this.CurrentStateTimer)
          }
        }).catch(error => {
          console.error(error);
          clearInterval(this.CurrentStateTimer); // 处理网络错误时也清除定时器
        });
      }, 250);
    },

    toggleDroneConnection() {
      if (this.isDroneConnected) {
        this.disconnectDrone();
        this.isShowImg2 = false;
        this.isDroneCameraOpen = false;
        this.isFaceTracking = false;
        this.isVisualizationEnabled = false;
      } else {
        this.connectDrone();
      }
    },

    connectDrone() {
      this.fullscreenLoading = true;
      this.$http.get('http://127.0.0.1:8000/connect_drone')
          .then(response => {
            if (response.data.status === 1) {
              this.fullscreenLoading = false;
              this.isDroneConnected = true;
              this.$message({
                message: '连接成功',
                type: 'success'
              });
              this.getCurrentState()

            } else {
              // 如果状态为 0，表示失败
              this.$message({
                message: data.message, // 显示后端返回的错误消息
                type: 'error'
              });
            }
            this.fullscreenLoading = false; // 结束加载动画
          })
          .catch(error => {
            this.fullscreenLoading = false; // 网络请求出错时结束加载动画
            console.error(error); // 打印错误日志
            this.$message({
              message: '网络请求失败，请检查后端服务是否正常运行',
              type: 'error'
            });
          });
    },

    disconnectDrone() {
      this.$http.get('http://127.0.0.1:8000/disconnect_drone')
          .then(response => {
            if (response.data.status === 1) {
              this.$message({
                message: response.data.message,
                type: 'success'
              });
            } else {
              this.$message({
                message: response.data.message,
                type: 'error'
              });
            }
            this.isDroneConnected = false;
          })
          .catch(error => {
            console.error(error);
            this.$message({
              message: '断开连接失败',
              type: 'error'
            });
          });
    },

    openVoice() {
      this.$http.get('http://127.0.0.1:8000/turn_voice')
          .then(response => {
            if (response.data.status === 1) {
              this.isVoice = true;
              this.$message.success(response.data.message);
            } else {
              this.isVoice = false;
              this.$message.error(response.data.message);
              this.voiceAction = "";
            }
            console.log(response.data, this.isVoice);
          })
          .catch(error => {
            console.error(error);
            this.$message.error('语音控制操作失败');
            this.voiceAction = ""; // 清除当前命令
          });
    },

    startProgress() {
      this.resetProgress(); // 先重置已有进度
      this.progressStatus = 'active'; // 新增状态标识
      // 模拟录音进度
      this.timer = setInterval(() => {
        if (this.progress < 100) {
          this.progress += 10; // 每次增加 5%
        } else {
          clearInterval(this.timer); // 停止定时器
          this.progressStatus = 'success'; // 录音完成
        }
      }, 200); // 每 200 毫秒更新一次
    },

    resetProgress() {
      if (this.timer) {
          clearInterval(this.timer);
          this.timer = null;
        }
        this.progress = 0;
        this.progressStatus = 'success';
    },

    recordVoice() {
      if (this.isVoice === false){
        this.$message({
          message: '请先开启语音控制',
          type: 'warning'
        });
        return;
      }

      // this.isRecordingVoice = true;

      if (!this.isRecordingVoice) {
        this.$message({
          message: '正在录音，请稍后',
          type: 'warning'
        });
      }
      else {
        return;
      }

      this.isRecordingVoice = true;
      this.startProgress(); // 启动进度条

      this.$http.get('http://127.0.0.1:8000/record_voice')
          .then(response => {
            if (response.data.status === 1) {
              this.$message({
                message: '识别成功',
                type: 'success'
              });
              this.voiceAction = response.data.command;
              // 强制将进度设为 100% 后再重置，确保视觉完整性
              this.progress = 100;
              setTimeout(() => this.resetProgress(), 200);
              this.isRecordingVoice = false;
            } else {
              this.$message({
                message: '识别失败',
                type: 'error'
              });
              this.voiceAction = "";
              this.progress = 0;
              setTimeout(() => this.resetProgress(), 200);
              this.isRecordingVoice = false;
            }
          })
          .catch(error => {
            console.error(error);
            this.$message({
              message: '识别请求失败',
              type: 'error'
            });
          })
    },

    // 前端按键按下控制无人机动作
    sendCommand(command) {
      this.$http.post('http://127.0.0.1:8000/drone_control', {command: command})
          .then(response => {
            console.log(response.data);
            if (response.data.status === 1) {
              this.$message({
                message: response.data.message,
                type: 'success'
              });
            } else {
              this.$message({
                message: response.data.message,
                type: 'error'
              });
            }
          })
          .catch(error => {
            console.error(error);
            this.$message({
              message: '命令发送失败',
              type: 'error'
            });
          });
    },

    updateSpeed() {
      const speed = this.move_speed;
      this.$http.post('http://127.0.0.1:8000/update_speed', {speed: speed})
          .then(response => {
            if (response.data.status === 1) {
              this.$message({
                message: response.data.message,
                type: 'success'
              });
            } else {
              this.$message({
                message: response.data.message,
                type: 'error'
              });
            }
          })
          .catch(error => {
            console.error(error);
            this.$message({
              message: '速度设置失败',
              type: 'error'
            });
          });
    },

    faceTrack() {
      this.$http.get('http://127.0.0.1:8000/turn_face_track')
          .then(response => {
            if (response.data.status === 1) {
              this.isFaceTracking = true;
              this.$message({
                message: response.data.message,
                type: 'success'
              });
            } else if (response.data.status === 0) {
              this.isFaceTracking = false;
              this.isVisualizationEnabled = false;
              this.$message({
                message: response.data.message,
                type: 'error'
              });
            }
          })
          .catch(error => {
            console.error(error);
            this.$message({
              message: '人脸跟随操作失败',
              type: 'error'
            });
          })
    },

    redirectToAdmin() {
      window.open("http://127.0.0.1:8000/admin/", "_blank");
    },

    change() {
      let data = {'weight': this.value}
      this.$http.post('http://127.0.0.1:8000/weight', data)
          .then(response => {
            console.log(response.data);
          })
          .catch(error => {
            console.error(error);
          });
    },
    open(link) {
      this.$electron.shell.openExternal(link)
    },

    openHand() {
      this.$http.get('http://127.0.0.1:8000/turn_hand')
          .then(response => {
            if (response.data.status === 1) {
              this.$data.isHand = true;
              this.$message({
                  message: response.data.message,
                  type: 'success'
                  }
              );
            } else {
              this.$data.isHand = false;
              this.$message({
                  message: response.data.message,
                  type: 'error'
                  }
              );
            }

            console.log(response.data, this.isHand);
          })
          .catch(error => {
            console.error(error);
            this.$message.error('手势检测操作失败');
          });
    },

    openCamera1() {
      this.$http.get('http://127.0.0.1:8000/turn_pc_camera')
          .then(response => {
            if (response.data.status === 1) { //打开
              this.isShowImg1 = true;
              this.isPcCameraOpen = true;
              this.$message({
                message: response.data.message,
                type: 'success'
              })
            } else {  // 关闭
              this.isShowImg1 = false;
              this.isPcCameraOpen = false;
              this.$message({
                message: response.data.message,
                type: 'error'
              })
            }
            console.log(response.data.status, this.isShowImg1);
          })
          .catch(error => {
            console.error(error);
          });
    },

    openCamera2() {
      this.$http.get('http://127.0.0.1:8000/turn_drone_camera')
          .then(response => {
            if (response.data.status === 1) { //打开
              this.isShowImg2 = true;
              this.isDroneCameraOpen = true;
              this.$message({
                message: response.data.message,
                type: 'success'
              })
            } else {  // 关闭
              this.isShowImg2 = false;
              this.isDroneCameraOpen = false;
              this.$message({
                message: response.data.message,
                type: 'error'
              })
            }
            console.log(response.data.status, this.isShowImg2);
          })
          .catch(error => {
            console.error(error);
          });
    },

    login() {
      if (!this.isPcCameraOpen) {
        this.$message({
          message: '请打开摄像头',
          type: 'error'
        });
        return;
      }

      this.$http.get('http://127.0.0.1:8000/login_get_frame_info')
          .then(response => {
            // 获取人脸信息
            const faceInfo = response.data;
            if (faceInfo.status === 0) {
              this.$message({
                message: faceInfo.message,
                type: 'error'
              });
              return;
            }
            if (faceInfo.status === 1) {
              if (faceInfo.face_count > 1) {
                this.$message({
                  message: '确保画面中只有一人',
                  type: 'error'
                });
                return;
              } else if (faceInfo.face_count === 0) {
                this.$message({
                  message: '未检测到人脸',
                  type: 'error'
                });
                return;
              } else if (faceInfo.face_exists) {
                this.$message({
                  message: '认证成功',
                  type: 'success'
                });
                this.isLogin = 1;
                return;
              } else {
                this.$message({
                  message: '人脸不存在',
                  type: 'error'
                });
                return;
              }
            }
          })
    },

    storageFace() {
      if (!this.isPcCameraOpen) {
        this.$message({
          message: '电脑摄像头未开启',
          type: 'error'
        });
        return;
      }

      // 发送请求获取当前画面中的人脸信息
      this.$http.get('http://127.0.0.1:8000/login_get_frame_info')
          .then(response => {
            // 获取人脸信息
            const faceInfo = response.data;
            if (faceInfo.status === 0) {
              this.$message({
                message: faceInfo.message,
                type: 'error'
              });
              return;
            }

            if (faceInfo.face_count > 1) {
              this.$message({
                message: '确保画面中只有一人',
                type: 'error'
              });
              return;
            } else if (faceInfo.face_count === 0) {
              this.$message({
                message: '未检测到人脸',
                type: 'error'
              });
              return;
            } else if (faceInfo.face_exists) {
              this.$message({
                message: '该人脸已经存在',
                type: 'error'
              });
              return;
            }

            // 如果一切正常，发送录入请求
            let data = {'name': this.name};
            this.$http.post('http://127.0.0.1:8000/storage_face', data)
                .then(response => {
                  console.log(response.data);
                  if (response.data.status === 1) {
                    this.$message({
                      message: response.data.message,
                      type: 'success'
                    });
                  } else {
                    this.$message({
                      message: response.data.message,
                      type: 'error'
                    });
                  }
                })
                .catch(error => {
                  console.error(error);
                  this.$message({
                    message: '人脸录入失败',
                    type: 'error'
                  });
                });
          })
          .catch(error => {
            console.error(error);
            this.$message({
              message: '获取人脸信息失败',
              type: 'error'
            });
          });
    },

    toggleVisualization() {
    this.$http.get('http://127.0.0.1:8000/toggle_visualization')
      .then(response => {
        if (response.data.status === 1) {
          this.isVisualizationEnabled = true;
          this.$message({
            message: response.data.message,
            type: 'success'
          });
        } else {
          this.isVisualizationEnabled = false;
          this.$message({
            message: response.data.message,
            type: 'error'
          });
        }
      });
    }
  }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css?family=Source+Sans+Pro');
body {
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
}
</style>

<style scoped>/* 提取公共样式 */
.status-item {
  font-weight: 900;
  margin-top: 2vh;
  display: flex;
  justify-content: space-evenly;
}

.label {
  width: 50%;
  text-align: center;
  margin-left: 10px;
}

.value {
  width: 50%;
  text-align: center;
}
</style>

<style>.button-group {
  font-weight: 900;
  margin-top: 2vh;
  display: flex;
  justify-content: space-evenly;
}
.custom-button {
  width: 35%;
  font-weight: 600;
}
</style>