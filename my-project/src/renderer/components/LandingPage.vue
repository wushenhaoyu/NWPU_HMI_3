<template>
  <div>
    <el-container>
  <el-main style="background-color: #e9eef3;border-radius: 1.5vw;">
    <div style="display: flex;">    
      <div style="width: 70%;">
        <div style="height: 100%;width: 100%;">
          <div style="text-align: center;font-size: larger;font-weight: 900;color: black;background-color: #e9eef3; height: 5vh;line-height: 5vh;">摄像头画面</div>
            <div v-if="isShowImg1 && isLogin != 2">            
              <img   :src="imgurl1" alt="electron-vue" >
            </div>
            <div v-if="!isShowImg1 && isLogin != 2">
              <img  style="height: 60%;width: 95%;object-fit: contain;" src="~@/assets/bg.png" alt="electron-vue" >
            </div>
            <div v-if="isShowImg2 && isLogin == 2">
              <img   :src="imgurl2" alt="electron-vue" >
            </div>
            <div v-if="!isShowImg2 && isLogin == 2">
              <img  style="height: 60%;width: 95%;object-fit: contain;" src="~@/assets/bg.png" alt="electron-vue" >
            </div>
          </div>


      </div>
      <div style="width: 30%;height: 90vh;"><div style="height: 4vh;"></div><div style="height: 86vh;background-color: #ffffff;border-radius: 1.5vw;">
        <div v-if="isLogin == 1"><!---->
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

                   <!---------------------------------------------------------->

        <div v-if="value1==1">


                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                     @click="openPoint">
                  <div style="width: 40%;text-align: center;">关键点显示</div>
                  <el-switch
                      v-model="isPoint"
                      active-text="开启"
                      inactive-text="关闭">
                  </el-switch>
                </div>


          <div style="height: 20vh;"></div>
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-input style="width: 80%;" v-model="name" placeholder="请输入名字"></el-input>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="storageFace">录入人脸</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera1">开启摄像头</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="redirectToAdmin" >人脸管理</el-button>
        </div>
        </div>


           <!---------------------------------------------------------->
        
              <div v-if="value1==2">
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                     @click="openHand">
                  <div style="width: 40%;text-align: center;">手势识别</div>
                  <el-switch
                      v-model="isHand"
                      active-text="开启"
                      inactive-text="关闭">
                  </el-switch>
                </div>
                <!---<div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                     @click="openHandPoint">
                  <div style="width: 40%;text-align: center;">关键点显示</div>
                  <el-switch
                      v-model="isHandPoint"
                      active-text="开启"
                      inactive-text="关闭">
                  </el-switch>
                </div>

                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                     @click="openPoint">
                  <div style="width: 40%;text-align: center;">手势控制</div>
                  <el-switch
                      v-model="isPoint"
                      active-text="开启"
                      inactive-text="关闭">
                  </el-switch>
                </div>-->
              </div>

              <!---------------------------------------------------------->
          <div v-if="value1==3">
          
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">电池电量：</div>
            <div style="width: 32%;text-align: center;">{{currentState.bat}}%</div>
          </div>
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">飞行高度：</div>
            <div style="width: 32%;text-align: center;">{{currentState.h}}cm</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">飞行时间：</div>
            <div style="width: 32%;text-align: center;">{{currentState.time}}s</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">信号强度：</div>
            <div style="width: 32%;text-align: center;">{{currentState.wifi}}s</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">加速度：</div>
            <div style="width: 32%;text-align: center;">({{currentState.agx}},{{currentState.agy}},{{currentState.agz}})</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">速度：</div>
            <div style="width: 32%;text-align: center;">({{currentState.vgx}},{{currentState.vgy}},{{currentState.vgz}})</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">航向角</div>
            <div style="width: 32%;text-align: center;">{{currentState.pitch}}°</div>
          </div>

          <div style="height: 8vh;"></div>




          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-button type="primary" style="width: 80%;font-weight: 600;" @click="wifiConnect"  v-loading.fullscreen.lock="fullscreenLoading">连接无人机</el-button>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-button type="primary" style="width: 80%;font-weight: 600;" @click="switchboard" >切换至无人机</el-button>
          </div>




          
          
        </div>
        <!---------------------------------------------------------->


      

        </div>

        <div v-if="isLogin == 2"><!---->
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

                   <!---------------------------------------------------------->

        <div v-if="value2==1">




          <div style="height: 40vh;"></div>
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera2">开启人脸跟随</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera">开启摄像头</el-button>
        </div>
        </div>
        

           <!---------------------------------------------------------->
        

              <!---------------------------------------------------------->
        <div v-if="value2==2">
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"
                     @click="openVoice">
                  <div style="width: 40%;text-align: center;">使能语音控制</div>
                  <el-switch
                      v-model="isVoice"
                      active-text="开启"
                      inactive-text="关闭">
                  </el-switch>
                </div>



          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;font-size: 4vh;"> <div style="width: 40%;text-align: center;">当前指令</div>{{ getVoiceAction }}</div>
          
          <div style="height: 35vh;"></div>
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: center;"
                     >
                     <el-progress style="width: 80%;" :percentage="progress" :status="progressStatus"  :text-inside="true" :stroke-width="26"/>
                </div>

  
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">

              <el-button type="primary" style="width: 80%;font-weight: 600;" @click="recordVoice" :disabled="!isVoice">开始录音</el-button>
           

          </div>
        </div>
           <!---------------------------------------------------------->

        <div v-if="value2==3">
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('takeoff')" >起飞</el-button>
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('land')" >降落</el-button>
                </div>

                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('up')" >上升</el-button>
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('down')" >下降</el-button>
                </div>
          
                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('forward')" >前进</el-button>
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('back')" >后退</el-button>
                </div>

                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('left')" >左移</el-button>
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('right')" >右移</el-button>
                </div>

                <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('rotate_left')" >左旋</el-button>
                  <el-button type="primary" style="width: 35%;font-weight: 600;" @click="sendCommand('rotate_right')" >右旋</el-button>
                </div>

                
            <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;line-height: 7vh;font-size: 2.5vh;"> 
              <div style="width: 35%;text-align: center;">移动速度cm/s</div><el-input-number v-model="move_speed"  :min="10" :max="20" label="描述文字"></el-input-number>
            </div>


            <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-button type="primary" style="width: 80%;font-weight: 600;" @click="recordVoice" >设置</el-button>
            </div>
        </div>
           <!---------------------------------------------------------->
        <div v-if="value2==4">
          
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">电池电量：</div>
            <div style="width: 32%;text-align: center;">{{currentState.bat}}%</div>
          </div>
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">飞行高度：</div>
            <div style="width: 32%;text-align: center;">{{currentState.h}}cm</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">飞行时间：</div>
            <div style="width: 32%;text-align: center;">{{currentState.time}}s</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">信号强度：</div>
            <div style="width: 32%;text-align: center;">{{currentState.wifi}}s</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">加速度：</div>
            <div style="width: 32%;text-align: center;">({{currentState.agx}},{{currentState.agy}},{{currentState.agz}})</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">速度：</div>
            <div style="width: 32%;text-align: center;">({{currentState.vgx}},{{currentState.vgy}},{{currentState.vgz}})</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <div style="width: 40%;text-align: center;">航向角</div>
            <div style="width: 32%;text-align: center;">{{currentState.pitch}}°</div>
          </div>

          <div style="height: 8vh;"></div>




          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-button type="primary" style="width: 80%;font-weight: 600;" @click="wifiConnect"  v-loading.fullscreen.lock="fullscreenLoading">连接无人机</el-button>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-button type="primary" style="width: 80%;font-weight: 600;" @click="switchboard" >切换至地面站</el-button>
          </div>

        </div>

      </div>



        <div v-if="isLogin == 0">
        <div style="text-align: center;font-weight: 900;line-height: 10vh;font-size: 25px;">系统登录</div>

        <div><img src="~@/assets/lg.png" alt="electron-vue" style="width: 80%;margin-left: 10%;"></div>
        <div style="height: 10vh;"></div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-input style="width: 80%;" v-model="name" placeholder="请输入名字"></el-input>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="storageFace">注册人脸</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="login">登录</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera1">开启摄像头</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="redirectToAdmin" >人脸管理</el-button>
        </div>
      </div>


      </div><!---->


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
    components: { SystemInformation },
    data () {
      return {
        //
        fullscreenLoading: false,
        isLogin: 2,
        EyeCount : 0,
        MouthCount : 0,
        HeadLeftCount : 0,
        HeadRightCount : 0,
        HeadShakeCount : 0,
        voiceAction: -1,
        isRecordingVoice: false,
        isVoice:false,
        isShowImg1:false,
        isShowImg2:false,
        isFace:false,
        isPoint:false,
        isAlign:false,
        isEye:false,
        isMouth:false,
        isHead:false,
        isGettingCount:false,
        isHand:false,
        isHandPoint:false,
        progress: 0,
        progressStatus: 'success',
        timer: null,
        CurrentStateTimer: null,
        name:'',
        imgurl1_:'http://localhost:8000/video',
        imgurl2_:'http://localhost:8000/video',
        timestamp: Date.now(),
        move_speed: 10,
        fly_height: 100,
        currentState:{
          pitch: 0,  
          roll: 0,   
          yaw: 0,    
          vgx: 0,    
          vgy: 0,    
          vgz: 0,    
          templ: 0,  
          temph: 0,  
          tof: 0,    
          h: 0,      
          bat: 0,  
          baro: 0.0, 
          time: 0,   
          agx: 0.0,  
          agy: 0.0,  
          agz: 0.0,  
          wifi: 0 , 
        },
        options1: [{
          value: 1,
          label: '人脸操作'
        }, {
          value: 2,
          label: '手势控制'
        },
        {
          value: 3,
          label: '无人机状态'
        }],
        options2: [{
          value: 1,
          label: '人脸操作'
        },
        {
          value: 2,
          label: '语音控制'
        },
        {
          value: 3,
          label: '键盘控制'
        },
        {
          value: 4,
          label: '无人机状态'
        }],
        value1: 1,
        value2: 1
      }
    },
    mounted () {
      //
    },
    computed: {
      //
      imgurl1(){
        return `${this.imgurl1_}?t=${this.timestamp}`;
      },
      imgurl2(){
        return `${this.imgurl2_}?t=${this.timestamp}`;
      },
      getVoiceAction(){
        if(this.voiceAction == 0){
          return "起飞"
      }else if(this.voiceAction == 1){
          return "降落"
      }else if(this.voiceAction == 2){
          return "前进"
      }else if(this.voiceAction == 3){
          return "后退"
        }else if(this.voiceAction == 4){
          return "降落"
        }else{
          return "无命令"
        }
      }

    },
    watch: {
      //
    },
    methods: {

      switchboard()
      {
          if(this.isLogin == 1)
          {
            this.isLogin = 2
          }else if(this.isLogin == 2)
          {
            this.isLogin = 1
          }
          console.log(this.isLogin)
      },
      getCurrentState(){
        this.CurrentStateTimer = setInterval(() => {
          this.$http.get('http://127.0.0.1:8000/get_current_state').then(response => {
            if (response.data.status == 1) {
              this.currentState.agx = response.data.tello_state.agx;
              this.currentState.agy = response.data.tello_state.agy;
              this.currentState.agz = response.data.tello_state.agz;
              this.currentState.bat = response.data.tello_state.bat;
              this.currentState.baro = response.data.tello_state.baro;
              this.currentState.h = response.data.tello_state.h;
              this.currentState.temph = response.data.tello_state.temph;
              this.currentState.templ = response.data.tello_state.templ;
              this.currentState.tof = response.data.tello_state.tof;
              this.currentState.time = response.data.tello_state.time;
              this.currentState.vgx = response.data.tello_state.vgx;
              this.currentState.vgy = response.data.tello_state.vgy;
              this.currentState.vgz = response.data.tello_state.vgz;
              this.currentState.pitch = response.data.tello_state.pitch;
              this.currentState.roll = response.data.tello_state.roll;
              this.currentState.yaw = response.data.tello_state.yaw;
              this.currentState.wifi = response.data.tello_state.wifi;
            } else {
              this.$message({
                message: '获取状态参数失败',
                type: 'error'
              });
              clearInterval(this.CurrentStateTimer) 
            }
          })
        }, 250);
      },
      wifiConnect()
      {
        this.fullscreenLoading = true;
        this.$http.get('http://127.0.0.1:8000/wifi_connect')
          .then(response => {
            if (response.data.status == 1) {
              this.fullscreenLoading = false;
              this.$message({
                message: '连接成功',
                type: 'success'
              });
              this.getCurrentState()

            } else {
              this.$message({
                message: '连接失败',
                type: 'error'
              });
            }
          })
          .catch(error => {
            this.fullscreenLoading = false;
            console.error(error);
          })
      },

      startProgress() {
      this.progress = 0; // 重置进度条
      this.progressStatus = 'success'; // 重置状态
      if (this.timer) {
        clearInterval(this.timer);
      }
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

      recordVoice()
      {
        if(this.isRecordingVoice) {
          this.$message({
            message: '正在录音，请稍后',
            type: 'warning'
          });
          return;
        }else{
          this.startProgress(); // 启动进度条
          this.isRecordingVoice = true;
          this.$http.get('http://127.0.0.1:8000/record_voice')
        .then(response => {
            if (response.data.status == 1) {
              this.$message({
                message: '识别成功',
                type: 'success'
              });
              this.voiceAction = response.data.data;
            } else {
              this.$message({
                message: '识别失败',
                type: 'error'
              });
            }

            this.isRecordingVoice = false;
            
          })

        }

      },
      // 前端按键按下控制无人机动作
      sendCommand(command) {
        this.$http.post('http://127.0.0.1:8000/key_input', { request_key: command })
            .then(response => {
                console.log(response.data);
                this.$message({
                    message: response.data.message,
                    type: 'success'
                });
            })
            .catch(error => {
                console.error(error);
                this.$message({
                    message: '命令发送失败',
                    type: 'error'
                });
            });
        },


      openVoice(){
        this.$http.get('http://127.0.0.1:8000/turn_voice')
        .then(response => {
            if (response.data.status == 1) {
              this.$data.isVoice = true;
            } else {
              this.$data.isVoice = false;
            }
            console.log(response.data, this.isVoice);
          })

      },

      redirectToAdmin() {
        window.open("http://127.0.0.1:8000/admin/", "_blank");
  },

    // turnOnGetCount() {
    //   this.isGettingCount = true;
    //   this.intervalId = setInterval(() => {
    //     this.getCount();
    //     if (!this.isGettingCount) {
    //       clearInterval(this.intervalId);
    //     }
    //   }, 200);
    // },
    // turnoffGetCount() {
    //   this.isGettingCount = false;
    // },
    // controlGetCount(sign) {
    //   if (sign) {
    //     console.log(this.isEye, this.isMouth, this.isHead);
    //     if ((this.isHead && !this.isMouth && !this.isEye) ||
    //         (!this.isHead && this.isMouth && !this.isEye) ||
    //         (!this.isHead && !this.isMouth && this.isEye)) {
    //
    //       this.turnOnGetCount();
    //     }
    //   } else {
    //     if (!this.isHead && !this.isMouth && !this.isEye) {
    //       this.turnoffGetCount();
    //       console.log("turnoffGetCount");
    //     }
    //   }
    //
    // },
    //
    // getCount() {
    //   this.$http.get('http://127.0.0.1:8000/get_count')
    //       .then(response => {
    //         console.log(response.data);
    //         this.EyeCount = response.data.EyeCount;
    //         this.MouthCount = response.data.MouthCount;
    //         this.HeadLeftCount = response.data.HeadLeftCount;
    //         this.HeadRightCount = response.data.HeadRightCount;
    //         this.HeadShakeCount = response.data.HeadShakeCount;
    //       })
    // },
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

    openPoint() {
      this.$http.get('http://127.0.0.1:8000/turn_point')
          .then(response => {
            if (response.data.status == 1) {
              this.$data.isPoint = true;
            } else {
              this.$data.isPoint = false;
            }
            console.log(response.data, this.isPoint);
          })
    },
    openAlign() {
      this.$http.get('http://127.0.0.1:8000/turn_align')
          .then(response => {
            if (response.data.status == 1) {
              this.$data.isAlign = true;
            } else {
              this.$data.isAlign = false;
            }
            console.log(response.data, this.isAlign);
          })
    },
    openFace() {
      this.$http.get('http://127.0.0.1:8000/turn_face')
          .then(response => {
            if (response.data.status == 1) {
              this.$data.isFace = true;
            } else {
              this.$data.isFace = false;
            }
            console.log(response.data, this.isFace);
          })
          .catch(error => {
            console.error(error);
          })
    },
    // openEye() {
    //   this.$http.get('http://127.0.0.1:8000/turn_eye')
    //       .then(response => {
    //         if (response.data.status == 1) {
    //           this.controlGetCount(1)
    //           this.$data.isEye = true;
    //         } else {
    //           this.$data.isEye = false;
    //           this.controlGetCount(0)
    //         }
    //         console.log(response.data, this.isEye);
    //       })
    // },
    // openMouth() {
    //   this.$http.get('http://127.0.0.1:8000/turn_mouth')
    //       .then(response => {
    //         if (response.data.status == 1) {
    //           this.controlGetCount(1)
    //           this.$data.isMouth = true;
    //         } else {
    //           this.$data.isMouth = false;
    //           this.controlGetCount(0)
    //         }
    //         console.log(response.data, this.isMouth);
    //       })
    // },
    // openHead() {
    //   this.$http.get('http://127.0.0.1:8000/turn_head')
    //       .then(response => {
    //         if (response.data.status == 1) {
    //           this.controlGetCount(1)
    //           this.$data.isHead = true;
    //         } else {
    //           this.$data.isHead = false;
    //           this.controlGetCount(0)
    //         }
    //         console.log(response.data, this.isHead);
    //       })
    // },
    openHand() {
      this.$http.get('http://127.0.0.1:8000/turn_hand')
          .then(response => {
            if (response.data.status == 1) {
              this.$data.isHand = true;
            } else {
              this.$data.isHand = false;
            }
            console.log(response.data, this.isHand);
          })
    },
    // openHandPoint() {
    //   this.$http.get('http://127.0.0.1:8000/turn_hand_point')
    //       .then(response => {
    //         if (response.data.status == 1) {
    //           this.$data.isHandPoint = true;
    //         } else {
    //           this.$data.isHandPoint = false;
    //         }
    //         console.log(response.data, this.isHandPoint);
    //       })
    // },
    // reset_count() {
    //   this.$http.get('http://127.0.0.1:8000/reset_count')
    //       .then(response => {
    //         console.log(response.data);
    //       })
    //       .catch(error => {
    //         console.error(error);
    //       });
    // },
    openCamera1() {
      this.$http.get('http://127.0.0.1:8000/turn_camera')
          .then(response => {
            if (response.data.status == 1) {
              this.isShowImg1 = true;
            } else {
              this.isShowImg1 = false;
            }
            console.log(response.data.status, this.isShowImg2);
          })
          .catch(error => {
            console.error(error);
          });
    },
    openCamera2() {
      this.$http.get('http://127.0.0.1:8000/turn_camera')
          .then(response => {
            if (response.data.status == 1) {
              this.isShowImg2 = true;
            } else {
              this.isShowImg2 = false;
            }
            console.log(response.data.status, this.isShowImg2);
          })
          .catch(error => {
            console.error(error);
          });
    },
    // storageFace() {
    //   this.$message({
    //     message: '请保证摄像头前只有一个人',
    //     type: 'warning'
    //   });
    //   if (!this.isAlign) {
    //     this.openAlign()
    //   }
    //   let data = {'name': this.name}
    //   this.$http.post('http://127.0.0.1:8000/storage_face', data)
    //       .then(response => {
    //         console.log(response.data);
    //       })
    //       .catch(error => {
    //         console.error(error);
    //       });
    // }
    login()
    {
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
              type: 'warning'
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
              type: 'info'
            });
            this.isLogin = 1;
            return;
          }else {
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
      // 检查是否开启人脸对齐
      const namePattern = /^[a-zA-Z_]+$/;
      if (!namePattern.test(this.name)) {
          this.$message({
              message: '人名只能包含英文和下划线',
              type: 'warning'
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
            type: 'warning'
          });
          return;
        } else if (faceInfo.face_count === 0) {
          this.$message({
            message: '未检测到人脸',
            type: 'warning'
          });
          return;
        } else if (faceInfo.face_exists) {
          this.$message({
            message: '该人脸已经存在',
            type: 'warning'
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
              message: '人脸录入成功',
              type: 'success'
            });
          } else {
            // 根据不同的 message 显示不同的提示
            if (response.data.message === '该名字已存在，请选择不同的名字') {
              this.$message({
                message: '该名字已存在，请选择不同的名字',
                type: 'warning'
              });
            } else {
              this.$message({
                message: '人脸录入失败',
                type: 'error'
              });
            }
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
