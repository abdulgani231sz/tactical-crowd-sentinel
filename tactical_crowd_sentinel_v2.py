"""
╔══════════════════════════════════════════════════════════════════╗
║       TACTICAL CROWD SENTINEL  v2.2  — FAST + FIXED             ║
║  ByteTrack · BEV · Heatmap · Fixed Detection · 25+ FPS RTX3050  ║
╚══════════════════════════════════════════════════════════════════╝
RUN:
    python tactical_crowd_sentinel_v2.py --source video.mp4
    python tactical_crowd_sentinel_v2.py --source 0
    python tactical_crowd_sentinel_v2.py --source 0 --model yolov8s.pt
"""

import cv2, numpy as np, argparse, time, math, threading, queue
import collections
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from filterpy.kalman import KalmanFilter
    HAS_KF = True
except ImportError:
    HAS_KF = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[WARN] ultralytics not found — demo mode")

try:
    import torch
    _DEV   = "cuda" if torch.cuda.is_available() else "cpu"
    _HALF  = torch.cuda.is_available()
except ImportError:
    _DEV, _HALF = "cpu", False

print(f"[INFO] Device={_DEV.upper()}  FP16={_HALF}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Cfg:
    model:       str   = "yolov8n.pt"
    imgsz:       int   = 416          # 416 sweet-spot for RTX 3050
    conf:        float = 0.35
    iou:         float = 0.45
    device:      str   = ""
    infer_every: int   = 2            # infer 1 in N frames

    max_age:     int   = 35
    min_hits:    int   = 1            # confirm on 1st hit (faster vis)
    max_dpx:     int   = 140

    trail_len:   int   = 30
    heat_every:  int   = 1            # add heat every frame
    heat_radius: int   = 20
    heat_decay:  float = 0.95
    heat_alpha:  float = 0.40

    bev_w: int = 220
    bev_h: int = 220
    bev_every: int = 3

    panel_w:     int   = 290
    panel_every: int   = 2

    speed_run:   float = 15.0
    density_hi:  int   = 12
    group_min:   int   = 4
    group_r:     int   = 85
    surge_win:   int   = 6

C = Cfg()
C.device = _DEV


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════════════════════════
P = dict(
    bg      =(10,12,20),   grid  =(28,38,52),
    green   =(0,210,75),   amber =(0,175,250),
    red     =(30,30,235),  cyan  =(195,215,0),
    white   =(215,225,238),dim   =(75,88,98),
    panel_bg=(8,10,17),    bev_bg=(5,7,13),
)
TC_COL = dict(WHITE=(215,225,238),ALPHA=(0,210,75),
              BRAVO=(0,175,250),CHARLIE=(30,30,235))


def txt(img,s,x,y,col,sc=0.40,th=1):
    f=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,s,(x,y),f,sc,(0,0,0),th+2,cv2.LINE_AA)
    cv2.putText(img,s,(x,y),f,sc,col,th,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  TRACK
# ══════════════════════════════════════════════════════════════════════════════
class Track:
    _nxt=1
    def __init__(self,bbox):
        self.id=Track._nxt; Track._nxt+=1
        self.bbox=bbox.copy()
        self.hits=1; self.tsu=0; self.age=0
        self.confirmed=(C.min_hits<=1)
        self.color=self._col(self.id)
        self.trail=collections.deque(maxlen=C.trail_len)
        self.spds =collections.deque(maxlen=12)
        self.prev =None
        self.kf   =self._kf()
        if self.kf:
            self.kf.x[0,0]=(bbox[0]+bbox[2])/2
            self.kf.x[1,0]=(bbox[1]+bbox[3])/2

    @staticmethod
    def _col(tid):
        np.random.seed(tid*31+7)
        h=int(np.random.randint(0,180))
        return tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[h,200,200]]]),cv2.COLOR_HSV2BGR)[0][0])

    def _kf(self):
        if not HAS_KF: return None
        kf=KalmanFilter(dim_x=4,dim_z=2)
        kf.F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],float)
        kf.H=np.array([[1,0,0,0],[0,1,0,0]],float)
        kf.R*=5; kf.P[2:,2:]*=100; kf.Q[2:,2:]*=0.5
        return kf

    def predict(self):
        if self.kf: self.kf.predict()
        self.age+=1; self.tsu+=1

    def update(self,bbox):
        self.bbox=bbox.copy()
        cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
        if self.kf:
            self.kf.update(np.array([[cx],[cy]],float))
            cx=float(np.squeeze(self.kf.x[0]))
            cy=float(np.squeeze(self.kf.x[1]))
        if self.prev is not None:
            self.spds.append(math.hypot(cx-self.prev[0],cy-self.prev[1]))
        self.prev=np.array([cx,cy])
        self.trail.append((int(cx),int(cy)))
        self.hits+=1; self.tsu=0
        if self.hits>=C.min_hits: self.confirmed=True

    @property
    def center(self):
        cx=(self.bbox[0]+self.bbox[2])/2; cy=(self.bbox[1]+self.bbox[3])/2
        if self.kf:
            cx=float(np.squeeze(self.kf.x[0]))
            cy=float(np.squeeze(self.kf.x[1]))
        return int(cx),int(cy)

    @property
    def foot(self):
        return int((self.bbox[0]+self.bbox[2])/2),int(self.bbox[3])

    @property
    def speed(self):
        return float(np.mean(self.spds)) if self.spds else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER  (simplified ByteTrack)
# ══════════════════════════════════════════════════════════════════════════════
def _iou(a,b):
    ix1=max(a[0],b[0]);iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]);iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

class Tracker:
    def __init__(self):
        self.tracks:List[Track]=[]
        self.lost  :List[Track]=[]

    def update(self,dets:np.ndarray)->List[Track]:
        # predict all
        for t in self.tracks+self.lost: t.predict()

        if len(dets)==0:
            self._cull(); return [t for t in self.tracks if t.confirmed]

        hi=dets[dets[:,4]>=C.conf]
        lo=dets[dets[:,4]< C.conf]

        matched_t,matched_d=self._assign(self.tracks,hi)

        # unmatched hi → new tracks
        for j in range(len(hi)):
            if j not in matched_d:
                self.tracks.append(Track(hi[j,:4]))

        # try lo against lost tracks
        live_lost=[t for t in self.lost if t.tsu<=C.max_age//2]
        self._assign(live_lost,lo)

        self._cull()
        return [t for t in self.tracks if t.confirmed]

    def _assign(self,tracks,dets):
        matched_t=set(); matched_d=set()
        if not len(tracks) or not len(dets):
            return matched_t,matched_d

        # cost matrix: IoU + distance
        tc=np.array([t.center for t in tracks],float)
        dc=np.array([((d[0]+d[2])/2,(d[1]+d[3])/2) for d in dets],float)
        pairs=[]
        for i,t in enumerate(tracks):
            for j,d in enumerate(dets):
                iou_v=_iou(t.bbox,d[:4])
                dist =math.hypot(tc[i,0]-dc[j,0],tc[i,1]-dc[j,1])
                cost =(1-iou_v)+0.3*min(dist/C.max_dpx,2.0)
                pairs.append((cost,i,j))

        pairs.sort(key=lambda x:x[0])
        for cost,i,j in pairs:
            if cost>1.8: break
            if i in matched_t or j in matched_d: continue
            tracks[i].update(dets[j,:4])
            matched_t.add(i); matched_d.add(j)
        return matched_t,matched_d

    def _cull(self):
        self.lost +=[t for t in self.tracks if t.tsu>C.max_age]
        self.tracks=[t for t in self.tracks if t.tsu<=C.max_age]
        self.lost  =[t for t in self.lost   if t.tsu<=C.max_age*2]


# ══════════════════════════════════════════════════════════════════════════════
#  THREAT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Threat:
    score:int=0; level:str="WHITE"
    flags:List[str]=field(default_factory=list)
    runners:List[int]=field(default_factory=list)
    groups:List[List[int]]=field(default_factory=list)
    surge:bool=False; cflow:bool=False

def analyse(tracks,win)->Threat:
    r=Threat(); n=len(tracks)
    if not n: return r
    cens=np.array([t.center for t in tracks],float)

    for t in tracks:
        if t.speed>C.speed_run: r.runners.append(t.id); r.score+=15

    win.append(n)
    if len(win)>=C.surge_win:
        bl=np.mean(list(win)[:-C.surge_win//2])
        rc=np.mean(list(win)[-C.surge_win//2:])
        if bl>2 and (rc-bl)/bl>0.35: r.surge=True; r.score+=25

    if n>=C.group_min:
        vis=[False]*n
        for i in range(n):
            if vis[i]: continue
            g=[i]
            for j in range(i+1,n):
                if np.linalg.norm(cens[i]-cens[j])<C.group_r: g.append(j)
            if len(g)>=C.group_min:
                for k in g: vis[k]=True
                r.groups.append([tracks[k].id for k in g]); r.score+=10

    vecs=[]
    for t in tracks:
        if len(t.trail)>=6:
            dx=t.trail[-1][0]-t.trail[-6][0]; dy=t.trail[-1][1]-t.trail[-6][1]
            if abs(dx)+abs(dy)>3: vecs.append((dx,dy))
    if len(vecs)>=4:
        angs=sorted(math.atan2(v[1],v[0]) for v in vecs)
        if angs[-1]-angs[0]>math.radians(120): r.cflow=True; r.score+=20

    if n>C.density_hi: r.score+=20

    if r.runners:    r.flags.append(f"RUNNING x{len(r.runners)}")
    if r.surge:      r.flags.append("CROWD SURGE")
    if r.groups:     r.flags.append(f"GROUPS x{len(r.groups)}")
    if r.cflow:      r.flags.append("COUNTER FLOW")
    if n>C.density_hi: r.flags.append("HIGH DENSITY")

    s=r.score
    r.level="WHITE" if s<15 else "ALPHA" if s<30 else "BRAVO" if s<55 else "CHARLIE"
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAP  (vectorised, fast)
# ══════════════════════════════════════════════════════════════════════════════
class Heatmap:
    def __init__(self,w,h):
        self.W,self.H=w,h
        # half-res internal
        self.w=max(1,w//2); self.h=max(1,h//2)
        self.buf=np.zeros((self.h,self.w),np.float32)
        r=max(4,C.heat_radius//2)
        y,x=np.ogrid[-r:r+1,-r:r+1]
        k=np.exp(-(x*x+y*y)/(2*(r/2)**2))
        self._k=(k/k.max()*70).astype(np.float32)
        self._r=r

    def add(self,centers):
        r=self._r
        for (px,py) in centers:
            sx=int(px/2); sy=int(py/2)
            cx1=max(0,sx-r); cx2=min(self.w,sx+r+1)
            cy1=max(0,sy-r); cy2=min(self.h,sy+r+1)
            if cx2<=cx1 or cy2<=cy1: continue
            kx1=cx1-(sx-r); ky1=cy1-(sy-r)
            hs=self.buf[cy1:cy2,cx1:cx2]
            ks=self._k[ky1:ky1+(cy2-cy1),kx1:kx1+(cx2-cx1)]
            sh=min(hs.shape[0],ks.shape[0]); sw=min(hs.shape[1],ks.shape[1])
            self.buf[cy1:cy1+sh,cx1:cx1+sw]+=ks[:sh,:sw]

    def tick(self): self.buf*=C.heat_decay; np.clip(self.buf,0,255,out=self.buf)

    def render(self,frame):
        sm=self.buf.astype(np.uint8)
        col=cv2.applyColorMap(sm,cv2.COLORMAP_JET)
        col=cv2.resize(col,(self.W,self.H),interpolation=cv2.INTER_LINEAR)
        msk=cv2.resize(sm,(self.W,self.H),interpolation=cv2.INTER_LINEAR)
        alpha=(np.clip(msk,0,255).astype(np.float32)/255.0*C.heat_alpha)[:,:,None]
        np.clip(frame*(1-alpha)+col*alpha,0,255,out=frame,casting='unsafe')
        frame[:]=frame.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  BEV RADAR
# ══════════════════════════════════════════════════════════════════════════════
class BEV:
    def __init__(self,fw,fh):
        src=np.float32([[fw*.25,fh*.40],[fw*.75,fh*.40],
                         [fw*.95,fh],[fw*.05,fh]])
        dst=np.float32([[0,0],[C.bev_w,0],[C.bev_w,C.bev_h],[0,C.bev_h]])
        self.M=cv2.getPerspectiveTransform(src,dst)
        self.W=C.bev_w; self.H=C.bev_h
        self.th=np.zeros((self.H,self.W),np.float32)
        base=np.full((self.H,self.W,3),P["bev_bg"],np.uint8)
        for i in range(0,self.H,38): cv2.line(base,(0,i),(self.W,i),P["grid"],1)
        for i in range(0,self.W,38): cv2.line(base,(i,0),(i,self.H),P["grid"],1)
        cx,cy=self.W//2,self.H//2
        for r in [48,96,144]: cv2.circle(base,(cx,cy),r,P["grid"],1)
        self._base=base

    def project(self,feet):
        if not feet: return []
        pts=np.array(feet,np.float32).reshape(-1,1,2)
        out=cv2.perspectiveTransform(pts,self.M).reshape(-1,2)
        return [(int(np.clip(p[0],0,self.W-1)),int(np.clip(p[1],0,self.H-1))) for p in out]

    def render(self,tracks):
        c=self._base.copy(); self.th*=0.90
        feet=[t.foot for t in tracks]; mp=self.project(feet)
        r2=6
        for bx,by in mp:
            x1=max(0,bx-r2);x2=min(self.W,bx+r2+1)
            y1=max(0,by-r2);y2=min(self.H,by+r2+1)
            self.th[y1:y2,x1:x2]+=30
        np.clip(self.th,0,255,out=self.th)
        hc=cv2.applyColorMap(self.th.astype(np.uint8),cv2.COLORMAP_HOT)
        mk=(self.th>10).astype(np.uint8)[:,:,None]
        c=(c*(1-0.5*mk)+hc*0.5*mk).astype(np.uint8)
        for i,(bx,by) in enumerate(mp):
            cv2.circle(c,(bx,by),5,tracks[i].color,-1)
            cv2.circle(c,(bx,by),7,(255,255,255),1)
        cv2.rectangle(c,(0,0),(self.W-1,self.H-1),P["cyan"],1)
        txt(c,"BIRD'S EYE",4,9,P["cyan"],0.33)
        return c


# ══════════════════════════════════════════════════════════════════════════════
#  DRAW  TRACKS
# ══════════════════════════════════════════════════════════════════════════════
def draw(frame,tracks,rpt,fi):
    rids=set(rpt.runners); gids={tid for g in rpt.groups for tid in g}
    for t in tracks:
        pts=list(t.trail)
        for k in range(1,len(pts)):
            a=k/len(pts)
            cv2.line(frame,pts[k-1],pts[k],
                     tuple(int(v*a) for v in t.color),1,cv2.LINE_AA)
        x1,y1,x2,y2=map(int,t.bbox)
        bc=(P["red"] if t.id in rids else P["amber"] if t.id in gids else t.color)
        th2=3 if t.id in rids else 2
        cv2.rectangle(frame,(x1,y1),(x2,y2),bc,th2)
        lbl=f"#{t.id}"+(" RUN" if t.id in rids else "")
        lw,lh=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.33,1)[0]
        cv2.rectangle(frame,(x1,y1-lh-4),(x1+lw+3,y1),bc,-1)
        txt(frame,lbl,x1+2,y1-2,P["white"],0.33)
        bl=int(min(t.speed/C.speed_run,1.0)*(x2-x1))
        if bl>0:
            cv2.rectangle(frame,(x1,y2+2),(x1+bl,y2+4),
                          P["red"] if t.id in rids else P["green"],-1)
    for g in rpt.groups:
        gts=[t for t in tracks if t.id in g]
        if len(gts)<2: continue
        pts=np.array([t.center for t in gts],np.int32)
        cv2.polylines(frame,[cv2.convexHull(pts)],True,P["amber"],2,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL
# ══════════════════════════════════════════════════════════════════════════════
class Panel:
    def __init__(self,h):
        self.h=h; self.w=C.panel_w
        self.events=collections.deque(maxlen=30)
        self._cache=None

    def log(self,msg):
        self.events.appendleft(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

    def render(self,tracks,rpt,bev_img,fps,fi):
        if fi%C.panel_every!=0 and self._cache is not None:
            return self._cache
        p=np.full((self.h,self.w,3),P["panel_bg"],np.uint8)
        n=len(tracks); y=0

        # ── Header
        cv2.rectangle(p,(0,0),(self.w,26),(16,20,35),-1)
        txt(p,"CROWD SENTINEL v2",6,17,P["cyan"],0.46)
        y=32

        # ── Threatcon badge
        tc=TC_COL[rpt.level]
        pulse=(fi%20<10) if rpt.level in("BRAVO","CHARLIE") else True
        if pulse:
            cv2.rectangle(p,(4,y),(self.w-4,y+30),tc,-1)
            txt(p,f"THREATCON  {rpt.level}",10,y+20,P["bg"],0.54,2)
        y+=36

        # ── Score bar
        sw2=int(np.clip(rpt.score/80,0,1)*(self.w-16))
        cv2.rectangle(p,(8,y),(self.w-8,y+6),P["grid"],-1)
        bc2=(P["green"] if rpt.score<30 else P["amber"] if rpt.score<55 else P["red"])
        if sw2>0: cv2.rectangle(p,(8,y),(8+sw2,y+6),bc2,-1)
        txt(p,f"SCORE {rpt.score:.0f}",8,y+19,P["dim"],0.36)
        y+=26

        # ── Stats table
        cv2.line(p,(6,y),(self.w-6,y),P["grid"],1); y+=8
        def st(lb,v,col=P["white"]):
            nonlocal y
            txt(p,lb,8,y,P["dim"],0.34)
            txt(p,str(v),self.w-52,y,col,0.37)
            y+=16
        st("PEOPLE",n,P["red"] if n>C.density_hi else P["green"])
        st("FPS",f"{fps:.1f}")
        st("RUNNERS",len(rpt.runners),P["red"] if rpt.runners else P["green"])
        st("GROUPS",len(rpt.groups),P["amber"] if rpt.groups else P["green"])
        st("SURGE","YES" if rpt.surge else "NO",P["red"] if rpt.surge else P["green"])
        st("CFLOW","YES" if rpt.cflow else "NO",P["amber"] if rpt.cflow else P["green"])
        y+=3

        # ── Alerts
        cv2.line(p,(6,y),(self.w-6,y),P["grid"],1); y+=6
        txt(p,"ALERTS",6,y,P["dim"],0.33); y+=13
        for fl in (rpt.flags[:4] or ["ALL CLEAR"]):
            col2=P["green"] if fl=="ALL CLEAR" else P["red"]
            if fl!="ALL CLEAR":
                cv2.rectangle(p,(6,y-11),(self.w-6,y+2),(30,0,0),-1)
            txt(p,f"  {fl}",8,y,col2,0.36); y+=14
        y+=3

        # ── BEV
        cv2.line(p,(6,y),(self.w-6,y),P["grid"],1); y+=5
        txt(p,"RADAR",6,y,P["dim"],0.33); y+=8
        bh,bw2=bev_img.shape[:2]
        sf=(self.w-14)/bw2
        bd=cv2.resize(bev_img,(int(bw2*sf),int(bh*sf)))
        ph,pw2=bd.shape[:2]
        if y+ph<self.h: p[y:y+ph,7:7+pw2]=bd
        y+=ph+4

        # ── Event log
        if self.h-y>55:
            cv2.line(p,(6,y),(self.w-6,y),P["grid"],1); y+=6
            txt(p,"EVENT LOG",6,y,P["dim"],0.33); y+=12
            for ev in list(self.events)[:C.panel_w//14]:
                if y+12>self.h: break
                txt(p,ev,4,y,P["dim"],0.28); y+=11

        txt(p,datetime.now().strftime("%H:%M:%S"),4,self.h-4,P["dim"],0.28)
        self._cache=p
        return p


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE THREAD  (non-blocking)
# ══════════════════════════════════════════════════════════════════════════════
class InferThread(threading.Thread):
    def __init__(self,model):
        super().__init__(daemon=True)
        self.model=model
        self._q:queue.Queue=queue.Queue(maxsize=1)
        self._res=np.empty((0,5),np.float32)
        self._lock=threading.Lock()
        self._stop=threading.Event()
        self.ready=False      # ← flag: True once first result back

    def push(self,frame):
        try: self._q.put_nowait(frame.copy())
        except queue.Full: pass

    def get(self):
        with self._lock: return self._res.copy()

    def stop(self): self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try: frame=self._q.get(timeout=0.5)
            except queue.Empty: continue
            try:
                res=self.model.predict(
                    frame,imgsz=C.imgsz,conf=C.conf,iou=C.iou,
                    classes=[0],device=C.device,verbose=False,half=_HALF)[0]
                boxes=res.boxes.xyxy.cpu().numpy()
                confs=res.boxes.conf.cpu().numpy()[:,None]
                dets=np.hstack([boxes,confs]) if len(boxes) else np.empty((0,5),np.float32)
            except Exception as e:
                print(f"[WARN] {e}"); dets=np.empty((0,5),np.float32)
            with self._lock: self._res=dets
            self.ready=True


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC (demo)
# ══════════════════════════════════════════════════════════════════════════════
def synth(fi,W,H):
    np.random.seed(fi//3)
    n=np.random.randint(8,22); dets=[]
    for _ in range(n):
        cx=int(W*(0.1+0.8*np.random.rand())); cy=int(H*(0.35+0.5*np.random.rand()))
        w=np.random.randint(25,55); h=np.random.randint(55,110)
        dets.append([cx-w//2,cy-h//2,cx+w//2,cy+h//2,0.9])
    return np.array(dets,np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════
def run(source,model_path,headless,scale):
    model=None
    if HAS_YOLO and model_path:
        print(f"[INFO] Loading {model_path}...")
        try:
            model=YOLO(model_path)
            # Warmup — eliminates 1st-frame GPU spike
            dummy=np.zeros((C.imgsz,C.imgsz,3),np.uint8)
            model.predict(dummy,imgsz=C.imgsz,verbose=False,device=C.device,half=_HALF)
            print("[INFO] Warmed up ✓")
        except Exception as e:
            print(f"[WARN] {e} — demo mode"); model=None

    demo=(source in(None,"demo"))
    if demo:
        cap=None; fw,fh=1280,720
    else:
        src=int(source) if str(source).isdigit() else source
        cap=cv2.VideoCapture(src)
        if not cap.isOpened(): print(f"[ERROR] Cannot open {source}"); return
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hmap=Heatmap(fw,fh); bev=BEV(fw,fh)
    panel=Panel(fh); tracker=Tracker()
    count_w=collections.deque(maxlen=60)

    # Start inference thread
    infer_th=None
    if model:
        infer_th=InferThread(model); infer_th.start()

    if not headless:
        cv2.namedWindow("CROWD SENTINEL v2",cv2.WINDOW_NORMAL)
        dw=int((fw+C.panel_w)*scale); dh=int(fh*scale)
        cv2.resizeWindow("CROWD SENTINEL v2",dw,dh)

    fi=0; fps_buf=collections.deque(maxlen=30)
    tracks:List[Track]=[]; rpt=Threat()
    bev_cache=bev.render([]); prev_flags=set(); prev_lvl="WHITE"
    waiting_first=True   # waiting for first inference result

    print("[INFO] Running — press Q to quit")
    while True:
        t0=time.perf_counter()

        # ── Grab frame
        if cap:
            ret,frame=cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        else:
            frame=np.full((fh,fw,3),P["bg"],np.uint8)
            sd=synth(fi,fw,fh)
            for d in sd:
                x1,y1,x2,y2=map(int,d[:4])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(50,70,50),-1)

        # ── Push frame for inference every N frames
        if fi%C.infer_every==0:
            if infer_th:
                infer_th.push(frame)
                # Use latest result (non-blocking)
                if infer_th.ready or fi>10:
                    dets=infer_th.get()
                    waiting_first=False
                else:
                    dets=np.empty((0,5),np.float32)
            else:
                dets=synth(fi,fw,fh); waiting_first=False

            if not waiting_first:
                tracks=tracker.update(dets)
                rpt=analyse(tracks,count_w)
                nf=set(rpt.flags)
                for f in nf-prev_flags: panel.log(f)
                if rpt.level!="WHITE" and rpt.level!=prev_lvl:
                    panel.log(f"THREATCON {rpt.level}")
                prev_lvl=rpt.level; prev_flags=nf

        # ── BEV (throttled)
        if fi%C.bev_every==0:
            bev_cache=bev.render(tracks)

        # ── Heatmap
        hmap.add([t.center for t in tracks])
        hmap.tick()
        hmap.render(frame)

        # ── Tracks
        draw(frame,tracks,rpt,fi)

        # ── "INITIALIZING" overlay while first inference loads
        if waiting_first:
            cv2.rectangle(frame,(fw//2-140,fh//2-25),(fw//2+140,fh//2+25),(0,0,0),-1)
            txt(frame,"INITIALIZING MODEL...",fw//2-130,fh//2+8,P["cyan"],0.6,2)

        fps_buf.append(time.perf_counter()-t0)
        fps=1.0/(np.mean(fps_buf)+1e-9)

        panel_img=panel.render(tracks,rpt,bev_cache,fps,fi)
        composite=np.hstack([frame,panel_img])

        n=len(tracks)
        zc=P["green"] if n<C.density_hi//2 else P["amber"] if n<C.density_hi else P["red"]
        zl="LOW" if n<C.density_hi//2 else "MED" if n<C.density_hi else "HIGH"
        cv2.rectangle(composite,(0,0),(fw,21),(0,0,0),-1)
        txt(composite,
            f"DENSITY:{zl}  PEOPLE:{n}  THREATCON:{rpt.level}  FPS:{fps:.1f}",
            8,14,zc,0.44)

        if not headless:
            cv2.imshow("CROWD SENTINEL v2",composite)
            if cv2.waitKey(1)&0xFF==ord('q'): break

        fi+=1
        if demo and fi>600: print("[INFO] Demo done."); break

    if infer_th: infer_th.stop()
    if cap: cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",  default=None)
    ap.add_argument("--model",   default="yolov8n.pt")
    ap.add_argument("--device",  default=None)
    ap.add_argument("--imgsz",   type=int,   default=416)
    ap.add_argument("--scale",   type=float, default=1.0,
                    help="Window scale: 0.8 = smaller window")
    ap.add_argument("--headless",action="store_true")
    a=ap.parse_args()

    if a.device: C.device=a.device
    C.imgsz=a.imgsz
    if C.device=="cpu":
        C.imgsz=min(C.imgsz,320); C.infer_every=3
        print("[INFO] CPU: imgsz=320, infer_every=3")

    run(a.source or "demo", a.model, a.headless, a.scale)