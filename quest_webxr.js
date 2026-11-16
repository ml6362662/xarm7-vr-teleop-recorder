/* Genuine right-hand XRInputSource: trigger=0, primary=A=4, secondary=B=5.
 * No navigator.getGamepads(), keyboard substitution, or fabricated buttons.
 */
'use strict';
const startButton=document.getElementById('start'),status=document.getElementById('status');
const canvas=document.getElementById('canvas');
let session=null,reference=null,gl=null,program=null,buffer=null,matrixLocation=null;
let frameId='',sequence=0,inFlight=false,pending=null;
let previous={trigger:null,a:null,b:null},releases={trigger:0,a:0,b:0};
let clockOffset=null;
let xrMode='immersive-vr';startButton.disabled=true;
async function synchronizeClock(){
  const before=performance.now()/1000;
  const response=await fetch('/clock?token='+encodeURIComponent(token),{cache:'no-store',signal:AbortSignal.timeout(300)});
  if(!response.ok)throw new Error('时钟同步失败');
  const value=await response.json(),after=performance.now()/1000;
  if(after-before>.1)throw new Error('USB 延迟超过 100 ms，禁止使用过期输入');
  clockOffset=value.monotonic_s-(before+after)/2;
}
function id(){return 'xr-'+Date.now()+'-'+Math.random().toString(36).slice(2);}
function invalid(reason,ended=false){return {type:'quest_webxr_v1',valid:false,reason,ended,frame_id:frameId,sequence:sequence++};}
function post(packet){
  pending={packet,time:performance.now()};
  if(!inFlight)flush();
}
async function flush(){
  if(!pending)return;
  const item=pending;pending=null;inFlight=true;
  item.packet.sample_age_s=(performance.now()-item.time)/1000;
  try{
    const response=await fetch('/pose?token='+encodeURIComponent(token),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(item.packet),cache:'no-store',signal:AbortSignal.timeout(300)});
    if(!response.ok)throw new Error(await response.text());
  }catch(error){status.textContent='电脑连接失败：'+error.message;}
  finally{inFlight=false;if(pending)flush();}
}
function multiply(a,b){const c=new Float32Array(16);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)c[col*4+row]+=a[k*4+row]*b[col*4+k];return c;}
function shader(type,source){const result=gl.createShader(type);gl.shaderSource(result,source);gl.compileShader(result);if(!gl.getShaderParameter(result,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(result));return result;}
function initializeGL(){
  gl=canvas.getContext('webgl',{xrCompatible:true,antialias:true});if(!gl)throw new Error('WebGL 不可用');
  program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec3 p;attribute vec3 c;uniform mat4 m;varying vec3 color;void main(){gl_Position=m*vec4(p,1.0);color=c;}'));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 color;void main(){gl_FragColor=vec4(color,1.0);}'));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('渲染初始化失败');
  buffer=gl.createBuffer();matrixLocation=gl.getUniformLocation(program,'m');
}
function draw(frame,viewer,rightPose,trigger,b){
  const layer=session.renderState.baseLayer;gl.bindFramebuffer(gl.FRAMEBUFFER,layer.framebuffer);
  if(xrMode==='immersive-ar')gl.clearColor(0,0,0,0);else gl.clearColor(.03,.045,.08,1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const data=[];function line(a,z,c){data.push(...a,...c,...z,...c);}
  for(let n=-8;n<=8;n++){line([n*.25,0,-2],[n*.25,0,2],[.15,.2,.3]);line([-2,0,n*.25],[2,0,n*.25],[.15,.2,.3]);}
  if(rightPose){
    const m=rightPose.transform.matrix,p=[m[12],m[13],m[14]];
    const colors=[[1,.2,.2],[.2,1,.3],[.2,.5,1]];
    for(let axis=0;axis<3;axis++)line(p,p.map((v,i)=>v+m[axis*4+i]*.12),colors[axis]);
    const color=b?[1,.3,.2]:trigger?[.2,1,.5]:[.8,.85,1];
    for(let axis=0;axis<3;axis++){const a=p.slice(),z=p.slice();a[axis]-=.025;z[axis]+=.025;line(a,z,color);}
  }
  gl.useProgram(program);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);
  for(const [name,offset] of [['p',0],['c',12]]){const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,24,offset);}
  for(const view of viewer.views){const viewport=layer.getViewport(view);gl.viewport(viewport.x,viewport.y,viewport.width,viewport.height);gl.uniformMatrix4fv(matrixLocation,false,multiply(view.projectionMatrix,view.transform.inverse.matrix));gl.drawArrays(gl.LINES,0,data.length/6);}
}
function onFrame(time,frame){
  if(!session)return;
  session.requestAnimationFrame(onFrame);
  let pose=null,trigger=false,b=false,packet;
  try{
    const right=Array.from(session.inputSources).find(source=>source.handedness==='right'&&source.gripSpace&&source.gamepad);
    if(!right)throw new Error('等待右手柄；请保持摄像头能追踪');
    const gamepad=right.gamepad;
    if(gamepad.mapping!=='xr-standard'||gamepad.buttons.length<6)throw new Error('当前手柄未提供独立 A/B 按键，禁止猜测映射');
    const knownProfile=right.profiles.some(p=>p.startsWith('oculus-touch')||p.startsWith('meta-quest-touch'));
    if(!knownProfile)throw new Error('非 Quest Touch 配置，请核对按键映射：'+right.profiles.join(','));
    pose=frame.getPose(right.gripSpace,reference);if(!pose)throw new Error('右手柄追踪无效');
    if(session.visibilityState!=='visible')throw new Error('VR 场景非前台；输入已暂停');
    trigger=gamepad.buttons[0].pressed;b=gamepad.buttons[5].pressed;
    const buttons={trigger,a:gamepad.buttons[4].pressed,b};
    for(const key of Object.keys(buttons)){if(previous[key]===true&&buttons[key]===false)releases[key]++;previous[key]=buttons[key];}
    const p=pose.transform.position,q=pose.transform.orientation;
    // WebXR RH (forward -Z) -> existing Quest/Unity LH (forward +Z).
    if(clockOffset===null)throw new Error('等待电脑时钟同步');
    packet={type:'quest_webxr_v1',valid:true,frame_id:frameId,sequence:sequence++,sample_server_s:performance.now()/1000+clockOffset,position_m:[p.x,p.y,-p.z],quaternion_wxyz:[q.w,-q.x,-q.y,q.z],
      trigger_pressed:trigger,a_pressed:buttons.a,b_pressed:b,release_counters:{...releases}};
    status.textContent='右手柄追踪正常\nXYZ m：'+packet.position_m.map(x=>x.toFixed(4)).join(' / ')+'\n食指扳机：'+(trigger?'按下':'松开')+' · B：'+(b?'按下 / 闭爪':'松开 / 开爪')+'\n'+right.profiles.join(', ');
  }catch(error){packet=invalid(error.message);status.textContent=error.message;previous={trigger:null,a:null,b:null};}
  // Publish each XR frame. A reset-to-now 60-Hz gate quantizes a 90-Hz
  // headset to 45 Hz. The single in-flight/latest-pending slot bounds backlog.
  post(packet);
  const viewer=frame.getViewerPose(reference);if(viewer)draw(frame,viewer,pose,trigger,b);
}
startButton.onclick=async()=>{
  if(session)return;
  startButton.disabled=true;frameId=id();sequence=0;releases={trigger:0,a:0,b:0};previous={trigger:null,a:null,b:null};
  try{
    // requestSession is directly in the controller/user click handler.
    const xr=await navigator.xr.requestSession(xrMode,{optionalFeatures:['local-floor']});session=xr;
    initializeGL();await gl.makeXRCompatible();
    xr.updateRenderState({baseLayer:new XRWebGLLayer(xr,gl)});
    try{reference=await xr.requestReferenceSpace('local-floor');}catch(_){reference=await xr.requestReferenceSpace('local');}
    await synchronizeClock();
    reference.addEventListener('reset',()=>{post(invalid('坐标基准已改变，请在电脑端重新检查方向'));frameId=id();sequence=0;previous={trigger:null,a:null,b:null};releases={trigger:0,a:0,b:0};});
    xr.addEventListener('visibilitychange',()=>{if(xr.visibilityState!=='visible')post(invalid('头显场景失去前台，停止输入'));});
    xr.addEventListener('end',()=>{post(invalid('VR 采集已结束',true));session=null;startButton.disabled=false;status.textContent='已停止；再次点击开始采集';});
    xr.requestAnimationFrame(onFrame);
  }catch(error){if(session){await session.end().catch(()=>{});session=null;}startButton.disabled=false;status.textContent='启动失败：'+error.message+'\n请使用 Quest 浏览器，保持 USB 连接，并允许 VR 权限。';}
};
window.addEventListener('pagehide',()=>{if(frameId)post(invalid('采集页面关闭',true));});
if(!window.isSecureContext||!navigator.xr){status.textContent='需要 Quest 浏览器及 localhost 安全上下文；请用电脑 GUI 打开头显采集。';startButton.disabled=true;}
else (async()=>{
  const ar=await navigator.xr.isSessionSupported('immersive-ar');
  const vr=ar||await navigator.xr.isSessionSupported('immersive-vr');
  xrMode=ar?'immersive-ar':'immersive-vr';startButton.disabled=!vr;
  startButton.textContent=ar?'开始手柄采集（MR 透视）':'开始手柄采集（VR）';
  status.textContent=vr?(ar?'已就绪；优先透视显示现场，不上传摄像头图像。':'已就绪；VR非透视，试动时需旁站观察/急停。')+'\n请点击开始。此页面不会自动开启机械臂。':'当前浏览器不支持 WebXR';
})().catch(e=>{status.textContent=e.message;});
