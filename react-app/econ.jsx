import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, RotateCcw, StepForward, FastForward, ChevronLeft, ChevronRight, TrendingUp, Sigma } from "lucide-react";

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function formatNum(x, d = 2) {
    return Number.isFinite(x) ? x.toFixed(d) : "0.00";
}

function priceKey(p) {
    return Number(p).toFixed(3);
}

function demand1(p1, p2, k1, k2) {
    return Math.max(0, k1 - p1 + k2 * p2);
}

function demand2(p1, p2, k1, k2) {
    return Math.max(0, k1 - p2 + k2 * p1);
}

function profit1(p1, p2, c, k1, k2) {
    return (p1 - c) * demand1(p1, p2, k1, k2);
}

function profit2(p1, p2, c, k1, k2) {
    return (p2 - c) * demand2(p1, p2, k1, k2);
}

function calculatePrices(k1, k2, c, m) {
    const pe = (k1 + c) / (2 - k2);
    const pc = (2 * k1 + 2 * c * (1 - k2)) / (4 * (1 - k2));
    const piE = profit1(pe, pe, c, k1, k2);
    const piC = profit1(pc, pc, c, k1, k2);

    const start = 2 * pe - pc;
    const end = 2 * pc - pe;
    const prices = [];
    for (let i = 0; i < m; i += 1) {
        const t = i / Math.max(1, m - 1);
        prices.push(Number((start + (end - start) * t).toFixed(3)));
    }

    return { prices, pe, pc, piE, piC };
}

function buildStateMaps(prices) {
    const states = [];
    const byPair = new Map();
    let idx = 0;
    for (const p1 of prices) {
        for (const p2 of prices) {
            const state = { idx, p1, p2, label: `s(${p1.toFixed(1)},${p2.toFixed(1)})` };
            states.push(state);
            byPair.set(`${priceKey(p1)}|${priceKey(p2)}`, idx);
            idx += 1;
        }
    }
    return { states, byPair };
}

function stateIndex(p1, p2, byPair) {
    return byPair.get(`${priceKey(p1)}|${priceKey(p2)}`) ?? 0;
}

function makeQTable(nStates, nActions) {
    return Array.from({ length: nStates }, () => Array.from({ length: nActions }, () => 0));
}

function deepCopyQ(q) {
    return q.map((row) => [...row]);
}

function argmaxDet(row) {
    let best = 0;
    for (let i = 1; i < row.length; i += 1) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

function argmaxTieRandom(row) {
    const max = Math.max(...row);
    const ties = [];
    for (let i = 0; i < row.length; i += 1) {
        if (Math.abs(row[i] - max) < 1e-10) ties.push(i);
    }
    return ties[Math.floor(Math.random() * ties.length)];
}

function epsilonAt(step, beta) {
    return Math.exp(-beta * step);
}

function chooseAction(row, eps) {
    if (Math.random() < eps) {
        return { action: Math.floor(Math.random() * row.length), mode: "explore" };
    }
    return { action: argmaxTieRandom(row), mode: "exploit" };
}

function greedyPolicy(Q) {
    return Q.map((row) => argmaxDet(row));
}

function policiesEqual(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function normalizeStartPrice(startPrice, prices) {
    let best = prices[0];
    let minDist = Math.abs(startPrice - best);
    for (let i = 1; i < prices.length; i += 1) {
        const d = Math.abs(startPrice - prices[i]);
        if (d < minDist) {
            minDist = d;
            best = prices[i];
        }
    }
    return best;
}

function chooseStartPair(config, prices) {
    if (config.randomStartEachEpisode) {
        return {
            p1: prices[Math.floor(Math.random() * prices.length)],
            p2: prices[Math.floor(Math.random() * prices.length)],
        };
    }
    return {
        p1: normalizeStartPrice(config.fixedStartP1, prices),
        p2: normalizeStartPrice(config.fixedStartP2, prices),
    };
}

function runCoreStep(snapshot, config, env) {
    const stepCount = snapshot.stepCount + 1;
    const eps = epsilonAt(stepCount, config.beta);
    const s = snapshot.currentState;

    const row1 = snapshot.Q1[s];
    const row2 = snapshot.Q2[s];
    const picked1 = chooseAction(row1, eps);
    const picked2 = chooseAction(row2, eps);

    const p1Next = env.prices[picked1.action];
    const p2Next = env.prices[picked2.action];
    const sNext = stateIndex(p1Next, p2Next, env.byPair);

    const q1Val = demand1(p1Next, p2Next, config.k1, config.k2);
    const q2Val = demand2(p1Next, p2Next, config.k1, config.k2);
    const pi1Val = profit1(p1Next, p2Next, config.c, config.k1, config.k2);
    const pi2Val = profit2(p1Next, p2Next, config.c, config.k1, config.k2);

    const oldQ1 = snapshot.Q1[s][picked1.action];
    const oldQ2 = snapshot.Q2[s][picked2.action];
    const maxNextQ1 = Math.max(...snapshot.Q1[sNext]);
    const maxNextQ2 = Math.max(...snapshot.Q2[sNext]);
    const target1 = pi1Val + config.delta * maxNextQ1;
    const target2 = pi2Val + config.delta * maxNextQ2;
    const newQ1 = (1 - config.alpha) * oldQ1 + config.alpha * target1;
    const newQ2 = (1 - config.alpha) * oldQ2 + config.alpha * target2;

    const Q1 = deepCopyQ(snapshot.Q1);
    const Q2 = deepCopyQ(snapshot.Q2);
    Q1[s][picked1.action] = newQ1;
    Q2[s][picked2.action] = newQ2;

    let stableCount = snapshot.stableCount;
    let prevPolicy1 = snapshot.prevPolicy1;
    let prevPolicy2 = snapshot.prevPolicy2;
    let convergenceInfo = snapshot.convergenceInfo;

    if (stepCount % config.checkEvery === 0) {
        const curPi1 = greedyPolicy(Q1);
        const curPi2 = greedyPolicy(Q2);
        if (policiesEqual(curPi1, prevPolicy1) && policiesEqual(curPi2, prevPolicy2)) {
            stableCount += config.checkEvery;
            if (stableCount >= config.stableRequired) {
                convergenceInfo = {
                    converged: true,
                    periodsRun: stepCount,
                    stablePeriods: stableCount,
                    epsilonFinal: eps,
                };
            }
        } else {
            stableCount = 0;
            prevPolicy1 = curPi1;
            prevPolicy2 = curPi2;
        }
    }

    const update = {
        step: stepCount,
        stateLabel: env.states[s].label,
        nextStateLabel: env.states[sNext].label,
        a1Index: picked1.action,
        a2Index: picked2.action,
        a1Price: p1Next,
        a2Price: p2Next,
        mode1: picked1.mode,
        mode2: picked2.mode,
        reward1: pi1Val,
        reward2: pi2Val,
        demand1: q1Val,
        demand2: q2Val,
        oldQ1,
        oldQ2,
        maxNextQ1,
        maxNextQ2,
        newQ1,
        newQ2,
        eq1: `${formatNum(oldQ1, 4)} + ${formatNum(config.alpha, 3)} x (${formatNum(pi1Val, 4)} + ${formatNum(config.delta, 3)} x ${formatNum(maxNextQ1, 4)} - ${formatNum(oldQ1, 4)}) = ${formatNum(newQ1, 4)}`,
        eq2: `${formatNum(oldQ2, 4)} + ${formatNum(config.alpha, 3)} x (${formatNum(pi2Val, 4)} + ${formatNum(config.delta, 3)} x ${formatNum(maxNextQ2, 4)} - ${formatNum(oldQ2, 4)}) = ${formatNum(newQ2, 4)}`,
    };

    return {
        ...snapshot,
        Q1,
        Q2,
        currentState: sNext,
        currentP1: p1Next,
        currentP2: p2Next,
        stepCount,
        priceHistory: [...snapshot.priceHistory, { step: stepCount, p1: p1Next, p2: p2Next }],
        prevPolicy1,
        prevPolicy2,
        stableCount,
        convergenceInfo,
        lastUpdate: update,
    };
}

function makeCheckpoint(snapshot, eventType, note, extra = {}) {
    return {
        id: snapshot.history.length,
        eventType,
        note,
        stepCount: snapshot.stepCount,
        Q1: deepCopyQ(snapshot.Q1),
        Q2: deepCopyQ(snapshot.Q2),
        currentState: snapshot.currentState,
        currentP1: snapshot.currentP1,
        currentP2: snapshot.currentP2,
        priceHistory: [...snapshot.priceHistory],
        skippedSteps: [...snapshot.skippedSteps],
        convergenceInfo: snapshot.convergenceInfo,
        update: snapshot.lastUpdate,
        ...extra,
    };
}

function simulateOneStep(snapshot, config, env) {
    const next = runCoreStep(snapshot, config, env);
    const cp = makeCheckpoint(next, "step", `Step ${next.stepCount}`);
    return { ...next, history: [...next.history, cp] };
}


function buildFreshSnapshot(config, env) {
    const nStates = env.states.length;
    const nActions = env.prices.length;
    const Q1 = makeQTable(nStates, nActions);
    const Q2 = makeQTable(nStates, nActions);
    const start = chooseStartPair(config, env.prices);
    const s0 = stateIndex(start.p1, start.p2, env.byPair);

    const snapshot = {
        Q1,
        Q2,
        currentState: s0,
        currentP1: start.p1,
        currentP2: start.p2,
        stepCount: 0,
        priceHistory: [{ step: 0, p1: start.p1, p2: start.p2 }],
        skippedSteps: [],
        lastUpdate: null,
        prevPolicy1: greedyPolicy(Q1),
        prevPolicy2: greedyPolicy(Q2),
        stableCount: 0,
        convergenceInfo: null,
        history: [],
    };

    const cp = makeCheckpoint(snapshot, "init", "Initial state");
    snapshot.history = [cp];
    return snapshot;
}

function exportQTableCSV(Q, states, prices, filename) {
    const q = (s) => `"${s}"`;
    const header = [q("State"), ...prices.map((p) => q(`p=${p.toFixed(1)}`))].join(",");
    const rows = Q.map((row, idx) => [q(states[idx].label), ...row.map((v) => v.toFixed(6))].join(","));
    const csv = [header, ...rows].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function renderQRows(Q, states, prices) {
    return Q.map((row, idx) => {
        const best = argmaxDet(row);
        return (
            <tr key={`q-${idx}`} className={idx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                <td className="px-3 py-2 font-medium">{states[idx].label}</td>
                {row.map((v, j) => (
                    <td
                        key={`q-${idx}-${j}`}
                        className={[
                            "px-3 py-2 font-mono",
                            best === j ? "bg-emerald-50 text-emerald-700" : "",
                        ].join(" ")}
                    >
                        {formatNum(v)}
                    </td>
                ))}
            </tr>
        );
    });
}

function greedySuccessor(Q1, Q2, s, prices, byPair) {
    const a1 = argmaxDet(Q1[s]);
    const a2 = argmaxDet(Q2[s]);
    const p1 = prices[a1];
    const p2 = prices[a2];
    return { a1, a2, p1, p2, next: stateIndex(p1, p2, byPair) };
}

function followGreedyUntilLoop(Q1, Q2, startP1, startP2, env, maxSteps = 50000) {
    let s = stateIndex(startP1, startP2, env.byPair);
    const firstSeen = new Map();
    const path = [];

    for (let t = 0; t < maxSteps; t += 1) {
        if (firstSeen.has(s)) {
            const loopStart = firstSeen.get(s);
            return { path, loopStart, loop: path.slice(loopStart) };
        }
        firstSeen.set(s, t);
        const step = greedySuccessor(Q1, Q2, s, env.prices, env.byPair);
        path.push({
            t,
            state: env.states[s].label,
            a1Price: step.p1,
            a2Price: step.p2,
            nextState: env.states[step.next].label,
        });
        s = step.next;
    }

    return { path, loopStart: null, loop: [] };
}

function ParamSlider({ label, min, max, step, value, onChange, formatValue }) {
    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
                <Label>{label}</Label>
                <span className="font-mono text-slate-600">{formatValue ? formatValue(value) : value}</span>
            </div>
            <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} />
        </div>
    );
}

export default function EconomicsQLearningDemo() {
    const [k1, setK1] = useState(7);
    const [k2, setK2] = useState(0.5);
    const [c, setC] = useState(2);
    const [m, setM] = useState(7);
    const [alpha, setAlpha] = useState(0.125);
    const [delta, setDelta] = useState(0.95);
    const [beta, setBeta] = useState(0.00002);
    const [checkEvery, setCheckEvery] = useState(1000);
    const [stableRequired, setStableRequired] = useState(100000);
    const [maxPeriods, setMaxPeriods] = useState(2000000);
    const [randomStartEachEpisode, setRandomStartEachEpisode] = useState(true);
    const [fixedStartP1, setFixedStartP1] = useState(7);
    const [fixedStartP2, setFixedStartP2] = useState(7);
    const [batchSteps, setBatchSteps] = useState(1000);
    const [autoplay, setAutoplay] = useState(false);

    const [appliedConfig, setAppliedConfig] = useState({
        k1: 7,
        k2: 0.5,
        c: 2,
        m: 7,
        alpha: 0.125,
        delta: 0.95,
        beta: 0.00002,
        checkEvery: 1000,
        stableRequired: 100000,
        maxPeriods: 2000000,
        randomStartEachEpisode: true,
        fixedStartP1: 7,
        fixedStartP2: 7,
    });

    const pricing = useMemo(() => calculatePrices(appliedConfig.k1, appliedConfig.k2, appliedConfig.c, appliedConfig.m), [appliedConfig.k1, appliedConfig.k2, appliedConfig.c, appliedConfig.m]);
    const env = useMemo(() => {
        const maps = buildStateMaps(pricing.prices);
        return { ...maps, prices: pricing.prices };
    }, [pricing.prices]);

    const [Q1, setQ1] = useState(() => makeQTable(49, 7));
    const [Q2, setQ2] = useState(() => makeQTable(49, 7));
    const [currentState, setCurrentState] = useState(0);
    const [currentP1, setCurrentP1] = useState(7);
    const [currentP2, setCurrentP2] = useState(7);
    const [stepCount, setStepCount] = useState(0);
    const [priceHistory, setPriceHistory] = useState([{ step: 0, p1: 7, p2: 7 }]);
    const [skippedSteps, setSkippedSteps] = useState([]);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [prevPolicy1, setPrevPolicy1] = useState([]);
    const [prevPolicy2, setPrevPolicy2] = useState([]);
    const [stableCount, setStableCount] = useState(0);
    const [convergenceInfo, setConvergenceInfo] = useState(null);
    const [history, setHistory] = useState([]);
    const [playbackIndex, setPlaybackIndex] = useState(null);
    const [trajStartP1, setTrajStartP1] = useState(7);
    const [trajStartP2, setTrajStartP2] = useState(7);
    const [trajectory, setTrajectory] = useState(null);
    const [customPaths, setCustomPaths] = useState([{ id: 0, aliceInput: "", bobInput: "" }]);
    const [nextCustomId, setNextCustomId] = useState(1);

    const autoplayRef = useRef(null);
    const workerRef = useRef(null);
    const applySnapshotRef = useRef(null);
    const [workerBusy, setWorkerBusy] = useState(false);

    const hasPendingConfig = useMemo(() => {
        return (
            k1 !== appliedConfig.k1 ||
            k2 !== appliedConfig.k2 ||
            c !== appliedConfig.c ||
            m !== appliedConfig.m ||
            alpha !== appliedConfig.alpha ||
            delta !== appliedConfig.delta ||
            beta !== appliedConfig.beta ||
            checkEvery !== appliedConfig.checkEvery ||
            stableRequired !== appliedConfig.stableRequired ||
            maxPeriods !== appliedConfig.maxPeriods ||
            randomStartEachEpisode !== appliedConfig.randomStartEachEpisode ||
            fixedStartP1 !== appliedConfig.fixedStartP1 ||
            fixedStartP2 !== appliedConfig.fixedStartP2
        );
    }, [k1, k2, c, m, alpha, delta, beta, checkEvery, stableRequired, maxPeriods, randomStartEachEpisode, fixedStartP1, fixedStartP2, appliedConfig]);

    function buildCurrentSnapshot(overrides = {}) {
        return {
            Q1,
            Q2,
            currentState,
            currentP1,
            currentP2,
            stepCount,
            priceHistory,
            skippedSteps,
            lastUpdate,
            prevPolicy1,
            prevPolicy2,
            stableCount,
            convergenceInfo,
            history,
            ...overrides,
        };
    }

    function applySnapshot(snapshot, options = {}) {
        setQ1(snapshot.Q1);
        setQ2(snapshot.Q2);
        setCurrentState(snapshot.currentState);
        setCurrentP1(snapshot.currentP1);
        setCurrentP2(snapshot.currentP2);
        setStepCount(snapshot.stepCount);
        setPriceHistory(snapshot.priceHistory);
        setSkippedSteps(snapshot.skippedSteps);
        setLastUpdate(snapshot.lastUpdate);
        setPrevPolicy1(snapshot.prevPolicy1);
        setPrevPolicy2(snapshot.prevPolicy2);
        setStableCount(snapshot.stableCount);
        setConvergenceInfo(snapshot.convergenceInfo);
        setHistory(snapshot.history);
        if (options.resetPlayback) setPlaybackIndex(null);
    }

    const playbackFrame = useMemo(() => {
        if (playbackIndex == null) {
            return {
                Q1,
                Q2,
                currentState,
                currentP1,
                currentP2,
                stepCount,
                priceHistory,
                skippedSteps,
                update: lastUpdate,
                convergenceInfo,
                note: "Live state",
            };
        }
        const cp = history[playbackIndex];
        return {
            Q1: cp.Q1,
            Q2: cp.Q2,
            currentState: cp.currentState,
            currentP1: cp.currentP1,
            currentP2: cp.currentP2,
            stepCount: cp.stepCount,
            priceHistory: cp.priceHistory,
            skippedSteps: cp.skippedSteps,
            update: cp.update ?? null,
            convergenceInfo: cp.convergenceInfo,
            note: cp.note,
        };
    }, [Q1, Q2, currentState, currentP1, currentP2, stepCount, priceHistory, skippedSteps, lastUpdate, convergenceInfo, history, playbackIndex]);

    // Keep applySnapshotRef current so the worker onmessage closure is never stale
    useEffect(() => {
        applySnapshotRef.current = applySnapshot;
    });

    // Initialize worker once — inlined as a Blob to avoid import.meta
    useEffect(() => {
        const workerSrc = `
function priceKey(p){return Number(p).toFixed(3);}
function demand1(p1,p2,k1,k2){return Math.max(0,k1-p1+k2*p2);}
function demand2(p1,p2,k1,k2){return Math.max(0,k1-p2+k2*p1);}
function profit1(p1,p2,c,k1,k2){return(p1-c)*demand1(p1,p2,k1,k2);}
function profit2(p1,p2,c,k1,k2){return(p2-c)*demand2(p1,p2,k1,k2);}
function formatNum(x,d=2){return Number.isFinite(x)?x.toFixed(d):"0.00";}
function stateIndex(p1,p2,byPair){return byPair.get(priceKey(p1)+"|"+priceKey(p2))??0;}
function deepCopyQ(q){return q.map(row=>[...row]);}
function argmaxDet(row){let b=0;for(let i=1;i<row.length;i++)if(row[i]>row[b])b=i;return b;}
function argmaxTieRandom(row){const mx=Math.max(...row);const t=[];for(let i=0;i<row.length;i++)if(Math.abs(row[i]-mx)<1e-10)t.push(i);return t[Math.floor(Math.random()*t.length)];}
function epsilonAt(step,beta){return Math.exp(-beta*step);}
function chooseAction(row,eps){if(Math.random()<eps)return{action:Math.floor(Math.random()*row.length),mode:"explore"};return{action:argmaxTieRandom(row),mode:"exploit"};}
function greedyPolicy(Q){return Q.map(row=>argmaxDet(row));}
function policiesEqual(a,b){if(!a||!b||a.length!==b.length)return false;for(let i=0;i<a.length;i++)if(a[i]!==b[i])return false;return true;}
const PRICE_HISTORY_CAP=100;
function runCoreStep(snapshot,config,env,opts={}){
  const stepCount=snapshot.stepCount+1;
  const eps=epsilonAt(stepCount,config.beta);
  const s=snapshot.currentState;
  const picked1=chooseAction(snapshot.Q1[s],eps);
  const picked2=chooseAction(snapshot.Q2[s],eps);
  const p1Next=env.prices[picked1.action];
  const p2Next=env.prices[picked2.action];
  const sNext=stateIndex(p1Next,p2Next,env.byPair);
  const q1Val=demand1(p1Next,p2Next,config.k1,config.k2);
  const q2Val=demand2(p1Next,p2Next,config.k1,config.k2);
  const pi1Val=profit1(p1Next,p2Next,config.c,config.k1,config.k2);
  const pi2Val=profit2(p1Next,p2Next,config.c,config.k1,config.k2);
  const oldQ1=snapshot.Q1[s][picked1.action];
  const oldQ2=snapshot.Q2[s][picked2.action];
  let maxNextQ1=-Infinity,maxNextQ2=-Infinity;
  const nr1=snapshot.Q1[sNext],nr2=snapshot.Q2[sNext];
  for(let i=0;i<nr1.length;i++){if(nr1[i]>maxNextQ1)maxNextQ1=nr1[i];if(nr2[i]>maxNextQ2)maxNextQ2=nr2[i];}
  const newQ1=(1-config.alpha)*oldQ1+config.alpha*(pi1Val+config.delta*maxNextQ1);
  const newQ2=(1-config.alpha)*oldQ2+config.alpha*(pi2Val+config.delta*maxNextQ2);
  const Q1=opts.mutate?snapshot.Q1:deepCopyQ(snapshot.Q1);
  const Q2=opts.mutate?snapshot.Q2:deepCopyQ(snapshot.Q2);
  Q1[s][picked1.action]=newQ1;Q2[s][picked2.action]=newQ2;
  let stableCount=snapshot.stableCount,prevPolicy1=snapshot.prevPolicy1,prevPolicy2=snapshot.prevPolicy2,convergenceInfo=snapshot.convergenceInfo;
  if(stepCount%config.checkEvery===0){
    const cp1=greedyPolicy(Q1),cp2=greedyPolicy(Q2);
    if(policiesEqual(cp1,prevPolicy1)&&policiesEqual(cp2,prevPolicy2)){
      stableCount+=config.checkEvery;
      if(stableCount>=config.stableRequired)convergenceInfo={converged:true,periodsRun:stepCount,stablePeriods:stableCount,epsilonFinal:eps};
    }else{stableCount=0;prevPolicy1=cp1;prevPolicy2=cp2;}
  }
  let newPriceHistory;
  if(opts.capHistory){const base=snapshot.priceHistory;const st=base.length>=PRICE_HISTORY_CAP?base.length-(PRICE_HISTORY_CAP-1):0;newPriceHistory=base.slice(st);newPriceHistory.push({step:stepCount,p1:p1Next,p2:p2Next});}
  else{newPriceHistory=[...snapshot.priceHistory,{step:stepCount,p1:p1Next,p2:p2Next}];}
  const update={step:stepCount,stateLabel:env.states[s].label,nextStateLabel:env.states[sNext].label,a1Index:picked1.action,a2Index:picked2.action,a1Price:p1Next,a2Price:p2Next,mode1:picked1.mode,mode2:picked2.mode,reward1:pi1Val,reward2:pi2Val,demand1:q1Val,demand2:q2Val,oldQ1,oldQ2,maxNextQ1,maxNextQ2,newQ1,newQ2,
    eq1:formatNum(oldQ1,4)+" + "+formatNum(config.alpha,3)+" x ("+formatNum(pi1Val,4)+" + "+formatNum(config.delta,3)+" x "+formatNum(maxNextQ1,4)+" - "+formatNum(oldQ1,4)+") = "+formatNum(newQ1,4),
    eq2:formatNum(oldQ2,4)+" + "+formatNum(config.alpha,3)+" x ("+formatNum(pi2Val,4)+" + "+formatNum(config.delta,3)+" x "+formatNum(maxNextQ2,4)+" - "+formatNum(oldQ2,4)+") = "+formatNum(newQ2,4),
  };
  return{...snapshot,Q1,Q2,currentState:sNext,currentP1:p1Next,currentP2:p2Next,stepCount,priceHistory:newPriceHistory,prevPolicy1,prevPolicy2,stableCount,convergenceInfo,lastUpdate:update};
}
function makeCheckpoint(snapshot,eventType,note,extra={}){
  return{id:snapshot.history.length,eventType,note,stepCount:snapshot.stepCount,Q1:deepCopyQ(snapshot.Q1),Q2:deepCopyQ(snapshot.Q2),currentState:snapshot.currentState,currentP1:snapshot.currentP1,currentP2:snapshot.currentP2,priceHistory:[...snapshot.priceHistory],skippedSteps:[...snapshot.skippedSteps],convergenceInfo:snapshot.convergenceInfo,update:snapshot.lastUpdate,...extra};
}
function simulateBatch(snapshot,config,env,stepsToRun){
  let next={...snapshot,Q1:deepCopyQ(snapshot.Q1),Q2:deepCopyQ(snapshot.Q2),priceHistory:[...snapshot.priceHistory],skippedSteps:[...snapshot.skippedSteps],history:snapshot.history};
  const start=snapshot.stepCount+1;
  for(let i=0;i<stepsToRun;i++){if(next.stepCount>=config.maxPeriods)break;next=runCoreStep(next,config,env,{mutate:true,capHistory:true});if(next.convergenceInfo?.converged)break;}
  const end=next.stepCount;
  if(end>=start)next.skippedSteps=[...next.skippedSteps,{start,end}];
  const cp=makeCheckpoint(next,"batch","Fast-forwarded "+Math.max(0,end-start+1)+" step(s)",{stepsRequested:stepsToRun});
  next.history=[...next.history,cp];return next;
}
function simulateUntilConvergence(snapshot,config,env){
  let next={...snapshot,Q1:deepCopyQ(snapshot.Q1),Q2:deepCopyQ(snapshot.Q2),priceHistory:[...snapshot.priceHistory],skippedSteps:[...snapshot.skippedSteps],history:snapshot.history};
  const start=snapshot.stepCount+1;
  while(next.stepCount<config.maxPeriods&&!next.convergenceInfo?.converged)next=runCoreStep(next,config,env,{mutate:true,capHistory:true});
  if(!next.convergenceInfo?.converged&&next.stepCount>=config.maxPeriods)next.convergenceInfo={converged:false,periodsRun:next.stepCount,stablePeriods:next.stableCount,epsilonFinal:epsilonAt(next.stepCount,config.beta)};
  const end=next.stepCount;
  if(end>=start)next.skippedSteps=[...next.skippedSteps,{start,end}];
  const cp=makeCheckpoint(next,"convergence",next.convergenceInfo?.converged?"Converged":"Reached max periods");
  next.history=[...next.history,cp];return next;
}
self.onmessage=function(e){
  const{type,snapshot,config,stepsToRun}=e.data;
  const raw=e.data.env;
  const env={prices:raw.prices,states:raw.states,byPair:new Map(Object.entries(raw.byPair))};
  let result;
  if(type==="batch")result=simulateBatch(snapshot,config,env,stepsToRun);
  else if(type==="convergence")result=simulateUntilConvergence(snapshot,config,env);
  const newCheckpoint=result.history[result.history.length-1];
  self.postMessage({type,result:{...result,history:[]},newCheckpoint});
};`;
        const blob = new Blob([workerSrc], { type: "application/javascript" });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url);
        URL.revokeObjectURL(url);
        worker.onmessage = (e) => {
            const { result, newCheckpoint } = e.data;
            applySnapshotRef.current(result, { resetPlayback: true });
            setHistory((h) => [...h, newCheckpoint]);
            setWorkerBusy(false);
        };
        workerRef.current = worker;
        return () => workerRef.current?.terminate();
    }, []);

    useEffect(() => {
        if (!autoplay) {
            if (autoplayRef.current) clearInterval(autoplayRef.current);
            autoplayRef.current = null;
            return;
        }
        autoplayRef.current = setInterval(() => {
            setPlaybackIndex(null);
            const next = simulateOneStep(buildCurrentSnapshot(), appliedConfig, env);
            applySnapshot(next);
        }, 350);
        return () => {
            if (autoplayRef.current) clearInterval(autoplayRef.current);
        };
    }, [autoplay, Q1, Q2, currentState, currentP1, currentP2, stepCount, priceHistory, skippedSteps, lastUpdate, prevPolicy1, prevPolicy2, stableCount, convergenceInfo, history, appliedConfig, env]);

    useEffect(() => {
        if (convergenceInfo?.converged && autoplay) {
            setAutoplay(false);
        }
    }, [convergenceInfo, autoplay]);

    function configure() {
        const nextConfig = {
            k1,
            k2: clamp(k2, 0, 0.99),
            c,
            m,
            alpha,
            delta,
            beta,
            checkEvery: Math.max(1, Math.round(checkEvery)),
            stableRequired: Math.max(1, Math.round(stableRequired)),
            maxPeriods: Math.max(100, Math.round(maxPeriods)),
            randomStartEachEpisode,
            fixedStartP1,
            fixedStartP2,
        };
        const nextPricing = calculatePrices(nextConfig.k1, nextConfig.k2, nextConfig.c, nextConfig.m);
        const nextEnv = { ...buildStateMaps(nextPricing.prices), prices: nextPricing.prices };

        setAppliedConfig(nextConfig);
        setTrajStartP1(nextPricing.prices[Math.floor(nextPricing.prices.length / 2)]);
        setTrajStartP2(nextPricing.prices[Math.floor(nextPricing.prices.length / 2)]);
        setTrajectory(null);
        applySnapshot(buildFreshSnapshot(nextConfig, nextEnv), { resetPlayback: true });
        setAutoplay(false);
    }

    function resetRun() {
        applySnapshot(buildFreshSnapshot(appliedConfig, env), { resetPlayback: true });
        setTrajectory(null);
        setAutoplay(false);
    }

    function stepOnce() {
        const next = simulateOneStep(buildCurrentSnapshot(), appliedConfig, env);
        applySnapshot(next, { resetPlayback: true });
    }

    function buildWorkerEnv() {
        return { prices: env.prices, states: env.states, byPair: Object.fromEntries(env.byPair) };
    }

    function fastForward() {
        if (workerBusy) return;
        setWorkerBusy(true);
        setAutoplay(false);
        setPlaybackIndex(null);
        // Strip history from snapshot — worker doesn't need it and it's expensive to clone
        const snapshot = { ...buildCurrentSnapshot(), history: [] };
        workerRef.current.postMessage({
            type: "batch",
            snapshot,
            config: appliedConfig,
            env: buildWorkerEnv(),
            stepsToRun: Math.max(1, Math.round(batchSteps)),
        });
    }

    function fastForwardUntilConvergence() {
        if (workerBusy) return;
        setWorkerBusy(true);
        setAutoplay(false);
        setPlaybackIndex(null);
        const snapshot = { ...buildCurrentSnapshot(), history: [] };
        workerRef.current.postMessage({
            type: "convergence",
            snapshot,
            config: appliedConfig,
            env: buildWorkerEnv(),
        });
    }

    function computeTrajectory() {
        const startP1 = normalizeStartPrice(trajStartP1, env.prices);
        const startP2 = normalizeStartPrice(trajStartP2, env.prices);
        setTrajStartP1(startP1);
        setTrajStartP2(startP2);
        const t = followGreedyUntilLoop(Q1, Q2, startP1, startP2, env, 50000);
        setTrajectory(t);
    }

    const displayedPricePath = useMemo(() => {
        const steps = playbackFrame.priceHistory ?? [];
        if (!steps.length) return "No prices yet.";
        const rows = [];
        for (let i = Math.max(0, steps.length - 16); i < steps.length; i += 1) {
            const it = steps[i];
            rows.push(`t=${it.step}: (${it.p1.toFixed(1)}, ${it.p2.toFixed(1)})`);
        }
        return rows.join("  |  ");
    }, [playbackFrame]);

    const trajSummary = useMemo(() => {
        if (!trajectory || trajectory.loopStart == null || !trajectory.loop.length) return null;
        const loop = trajectory.loop;
        let sumP1 = 0;
        let sumP2 = 0;
        let sumPi1 = 0;
        let sumPi2 = 0;
        for (const rec of loop) {
            sumP1 += rec.a1Price;
            sumP2 += rec.a2Price;
            sumPi1 += profit1(rec.a1Price, rec.a2Price, appliedConfig.c, appliedConfig.k1, appliedConfig.k2);
            sumPi2 += profit2(rec.a1Price, rec.a2Price, appliedConfig.c, appliedConfig.k1, appliedConfig.k2);
        }
        const avgP1 = sumP1 / loop.length;
        const avgP2 = sumP2 / loop.length;
        const avgPi1 = sumPi1 / loop.length;
        const avgPi2 = sumPi2 / loop.length;
        const denom = pricing.piC - pricing.piE;
        const norm1 = Math.abs(denom) < 1e-10 ? null : (avgPi1 - pricing.piE) / denom;
        const norm2 = Math.abs(denom) < 1e-10 ? null : (avgPi2 - pricing.piE) / denom;
        return { avgP1, avgP2, avgPi1, avgPi2, norm1, norm2, loopLength: loop.length, loopStart: trajectory.loopStart };
    }, [trajectory, appliedConfig, pricing]);

    return (
        <div className="min-h-screen bg-slate-50 p-4 md:p-6">
            <div className="mx-auto space-y-6">
                <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight">Q-Learning in a Two-Firm Pricing Game</h1>
                    <p className="max-w-5xl text-slate-600">
                        Two players repeatedly choose prices. State is the current pair (p1, p2), actions are next prices, and each player learns with its own Q-table using period profit as reward.
                    </p>
                </div>

                <div className="grid gap-6 md:grid-cols-[390px_1fr]">
                    <div className="space-y-6">
                        <Card className="rounded-2xl">
                            <CardHeader>
                                <CardTitle>Environment and Learning Parameters</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-5">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="space-y-2">
                                        <Label>k1</Label>
                                        <Input type="number" step="0.1" value={k1} onChange={(e) => setK1(Number(e.target.value))} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>k2</Label>
                                        <Input type="number" step="0.01" value={k2} onChange={(e) => setK2(Number(e.target.value))} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Marginal Cost c</Label>
                                        <Input type="number" step="0.1" value={c} onChange={(e) => setC(Number(e.target.value))} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Action Space Size m</Label>
                                        <select
                                            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                                            value={m}
                                            onChange={(e) => {
                                                const v = Number(e.target.value);
                                                if (Number.isNaN(v)) return;
                                                setM(v);
                                            }}
                                        >
                                            {[4, 7, 10, 13, 16].map((option) => (
                                                <option key={option} value={option}>
                                                    {option}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                <ParamSlider label="Learning Rate (alpha)" min={0} max={1} step={0.001} value={alpha} onChange={setAlpha} formatValue={(v) => v.toFixed(3)} />
                                <ParamSlider label="Discount Factor (delta)" min={0} max={0.999} step={0.001} value={delta} onChange={setDelta} formatValue={(v) => v.toFixed(3)} />
                                <ParamSlider label="Exploration Decay (beta)" min={0} max={0.001} step={0.000001} value={beta} onChange={setBeta} formatValue={(v) => v.toFixed(6)} />

                                <div className="grid grid-cols-3 gap-3">
                                    <div className="space-y-2">
                                        <Label>Check Every</Label>
                                        <Input type="number" value={checkEvery} onChange={(e) => setCheckEvery(Number(e.target.value))} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Stable Required</Label>
                                        <Input type="number" value={stableRequired} onChange={(e) => setStableRequired(Number(e.target.value))} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Max Periods</Label>
                                        <Input type="number" value={maxPeriods} onChange={(e) => setMaxPeriods(Number(e.target.value))} />
                                    </div>
                                </div>

                                <div className="rounded-xl border p-3 text-sm">
                                    <div><span className="font-semibold">Equilibrium price:</span> {formatNum(pricing.pe)}</div>
                                    <div><span className="font-semibold">Collusion price:</span> {formatNum(pricing.pc)}</div>
                                    <div><span className="font-semibold">Action space:</span> {pricing.prices.map((p) => p.toFixed(1)).join(", ")}</div>
                                </div>

                                <div className="flex items-center justify-between rounded-xl border p-3">
                                    <div>
                                        <Label>Random starting price pair</Label>
                                        <p className="text-xs text-slate-500">If off, training always starts from fixed p1 and p2.</p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => setRandomStartEachEpisode((v) => !v)}
                                        className={["inline-flex h-6 w-11 items-center rounded-full transition", randomStartEachEpisode ? "bg-slate-900" : "bg-slate-300"].join(" ")}
                                        aria-pressed={randomStartEachEpisode}
                                    >
                                        <span className={["inline-block h-5 w-5 rounded-full bg-white transition", randomStartEachEpisode ? "translate-x-5" : "translate-x-0.5"].join(" ")} />
                                    </button>
                                </div>

                                {!randomStartEachEpisode && (
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="space-y-2">
                                            <Label>Fixed p1</Label>
                                            <Input type="number" step="0.1" value={fixedStartP1} onChange={(e) => setFixedStartP1(Number(e.target.value))} />
                                        </div>
                                        <div className="space-y-2">
                                            <Label>Fixed p2</Label>
                                            <Input type="number" step="0.1" value={fixedStartP2} onChange={(e) => setFixedStartP2(Number(e.target.value))} />
                                        </div>
                                    </div>
                                )}

                                {hasPendingConfig && (
                                    <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                                        Parameters changed. Click Configure to rebuild the action/state space and reset training.
                                    </div>
                                )}

                                <Button className="w-full" onClick={configure} disabled={!hasPendingConfig}>
                                    Configure
                                </Button>
                            </CardContent>
                        </Card>

                        <Card className="rounded-2xl">
                            <CardHeader>
                                <CardTitle>Training Controls</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-3">
                                    <Button variant="outline" onClick={resetRun} disabled={workerBusy}>
                                        <RotateCcw className="mr-2 h-4 w-4" />
                                        Full Reset
                                    </Button>
                                    <Button onClick={stepOnce} disabled={workerBusy}>
                                        <StepForward className="mr-2 h-4 w-4" />
                                        Step
                                    </Button>
                                    <Button
                                        variant={autoplay ? "destructive" : "default"}
                                        onClick={() => setAutoplay((v) => !v)}
                                        disabled={workerBusy}
                                    >
                                        {autoplay ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                                        {autoplay ? "Stop" : "Autoplay"}
                                    </Button>
                                    <Button variant="secondary" onClick={fastForward} disabled={workerBusy}>
                                        <FastForward className="mr-2 h-4 w-4" />
                                        Fast Forward
                                    </Button>
                                </div>

                                <div className="flex items-center gap-2">
                                    <Label className="whitespace-nowrap">Batch Steps</Label>
                                    <Input type="number" value={batchSteps} onChange={(e) => setBatchSteps(Number(e.target.value))} disabled={workerBusy} />
                                </div>

                                <Button variant="secondary" className="w-full" onClick={fastForwardUntilConvergence} disabled={workerBusy}>
                                    Run Until Convergence
                                </Button>

                                {workerBusy && (
                                    <div className="flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-800">
                                        <svg className="h-4 w-4 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                                        </svg>
                                        Training in progress…
                                    </div>
                                )}

                                <div className="rounded-xl border p-3">
                                    <div className="mb-3 flex items-center justify-between">
                                        <div>
                                            <div className="font-medium">Playback Controls</div>
                                            <div className="text-xs text-slate-500">Inspect earlier checkpoints in this run.</div>
                                        </div>
                                        <Badge variant="outline">{playbackIndex == null ? "Live" : `Playback #${playbackIndex + 1}`}</Badge>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setPlaybackIndex((p) => (history.length === 0 ? null : p == null ? history.length - 1 : Math.max(0, p - 1)))}
                                            disabled={history.length === 0}
                                        >
                                            <ChevronLeft className="h-4 w-4" />
                                        </Button>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setPlaybackIndex((p) => (history.length === 0 ? null : p == null ? 0 : Math.min(history.length - 1, p + 1)))}
                                            disabled={history.length === 0}
                                        >
                                            <ChevronRight className="h-4 w-4" />
                                        </Button>
                                        <Button variant="secondary" size="sm" onClick={() => setPlaybackIndex(null)} disabled={playbackIndex == null}>
                                            Return to Live
                                        </Button>
                                        <div className="text-xs text-slate-500">{history.length} checkpoints</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    <div className="space-y-6">
                        <Card className="rounded-2xl">
                            <CardHeader>
                                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                    <CardTitle>Simulation State</CardTitle>
                                    <div className="flex flex-wrap items-center gap-2 text-sm">
                                        <Badge variant="secondary">Step: {playbackFrame.stepCount}</Badge>
                                        <Badge>{playbackFrame.note}</Badge>
                                        <Badge variant="outline">Current: ({formatNum(playbackFrame.currentP1, 1)}, {formatNum(playbackFrame.currentP2, 1)})</Badge>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid gap-3 md:grid-cols-3">
                                    <div className="rounded-xl border p-3">
                                        <div className="text-xs text-slate-500">Current state</div>
                                        <div className="font-semibold">{env.states[playbackFrame.currentState]?.label}</div>
                                    </div>
                                    <div className="rounded-xl border p-3">
                                        <div className="text-xs text-slate-500">Current epsilon</div>
                                        <div className="font-semibold">{formatNum(epsilonAt(Math.max(1, playbackFrame.stepCount), appliedConfig.beta), 6)}</div>
                                    </div>
                                    <div className="rounded-xl border p-3">
                                        <div className="text-xs text-slate-500">Policy stability</div>
                                        <div className="font-semibold">{stableCount.toLocaleString()} / {appliedConfig.stableRequired.toLocaleString()}</div>
                                    </div>
                                </div>

                                <div className="rounded-xl border p-3">
                                    <div className="mb-2 flex items-center gap-2 font-semibold">
                                        <TrendingUp className="h-4 w-4" />
                                        Price Path (latest section)
                                    </div>
                                    <div className="rounded-lg bg-slate-100 p-2 font-mono text-xs text-slate-700">{displayedPricePath}</div>
                                    {!!(playbackFrame.skippedSteps?.length) && (
                                        <div className="mt-2 text-xs text-slate-500">
                                            Skipped ranges: {playbackFrame.skippedSteps.map((r) => `[${r.start}-${r.end}]`).join(", ")}
                                        </div>
                                    )}
                                </div>

                                {playbackFrame.convergenceInfo && (
                                    <div
                                        className={[
                                            "rounded-xl border p-3 text-sm",
                                            playbackFrame.convergenceInfo.converged ? "border-emerald-200 bg-emerald-50 text-emerald-900" : "border-amber-200 bg-amber-50 text-amber-900",
                                        ].join(" ")}
                                    >
                                        {playbackFrame.convergenceInfo.converged
                                            ? `Converged after ${playbackFrame.convergenceInfo.periodsRun.toLocaleString()} steps.`
                                            : `Not converged by ${playbackFrame.convergenceInfo.periodsRun.toLocaleString()} steps.`}
                                        {" "}Stable periods: {playbackFrame.convergenceInfo.stablePeriods.toLocaleString()}, final epsilon: {formatNum(playbackFrame.convergenceInfo.epsilonFinal, 6)}.
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        <div className="grid gap-6 xl:grid-cols-2">
                            <Card className="rounded-2xl">
                                <CardHeader>
                                    <CardTitle>Bellman Update (Latest)</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    {!playbackFrame.update && <p className="text-sm text-slate-500">No update yet. Step once to compute Q-updates for both players.</p>}
                                    {playbackFrame.update && (
                                        <>
                                            <div className="grid grid-cols-2 gap-3 text-sm">
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">State</span><div className="text-lg font-semibold">{playbackFrame.update.stateLabel}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Next State</span><div className="text-lg font-semibold">{playbackFrame.update.nextStateLabel}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Alice Action</span><div className="text-lg font-semibold">{formatNum(playbackFrame.update.a1Price, 1)} ({playbackFrame.update.mode1})</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Bob Action</span><div className="text-lg font-semibold">{formatNum(playbackFrame.update.a2Price, 1)} ({playbackFrame.update.mode2})</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Reward pi1</span><div className="text-lg font-semibold">{formatNum(playbackFrame.update.reward1, 4)}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Reward pi2</span><div className="text-lg font-semibold">{formatNum(playbackFrame.update.reward2, 4)}</div></div>
                                            </div>

                                            <div className="rounded-xl bg-slate-900 p-3 font-mono text-xs text-slate-100">
                                                <div className="mb-1">Q1: {playbackFrame.update.eq1}</div>
                                                <div>Q2: {playbackFrame.update.eq2}</div>
                                            </div>
                                        </>
                                    )}
                                </CardContent>
                            </Card>

                            <Card className="rounded-2xl">
                                <CardHeader>
                                    <CardTitle>Greedy Trajectory and Cycle</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <div className="grid grid-cols-2 gap-2">
                                        <Input type="number" step="0.1" value={trajStartP1} onChange={(e) => setTrajStartP1(Number(e.target.value))} />
                                        <Input type="number" step="0.1" value={trajStartP2} onChange={(e) => setTrajStartP2(Number(e.target.value))} />
                                    </div>
                                    <Button className="w-full" onClick={computeTrajectory}>
                                        <Sigma className="mr-2 h-4 w-4" />
                                        Compute Greedy Trajectory
                                    </Button>

                                    {!trajectory && <div className="text-sm text-slate-500">Compute trajectory to inspect long-run loop behavior under greedy play.</div>}
                                    {trajectory && (
                                        <div className="space-y-3 text-sm">
                                            <div className="rounded-xl border p-3">
                                                Path length: {trajectory.path.length.toLocaleString()}
                                                {trajectory.loopStart == null ? " | no loop found (within max steps)." : ` | loop starts at step ${trajectory.loopStart}, length ${trajectory.loop.length}`}
                                            </div>

                                            {/* Price path table — step 0 is the starting state, steps 1..N are actions taken */}
                                            {(() => {
                                                const fullPath = [
                                                    { a1Price: trajStartP1, a2Price: trajStartP2 },
                                                    ...trajectory.path.map((rec) => ({ a1Price: rec.a1Price, a2Price: rec.a2Price })),
                                                ];
                                                // loopStart in trajectory.path is 0-indexed into path; in fullPath it's offset by 1
                                                const loopStartFull = trajectory.loopStart != null ? trajectory.loopStart + 1 : null;
                                                return (
                                                    <div>
                                                        <div className="mb-1 font-medium text-slate-700">Price path (p1, p2)</div>
                                                        <div className="overflow-x-auto rounded-xl border">
                                                            <table className="border-collapse text-xs" style={{ tableLayout: "fixed" }}>
                                                                <tbody>
                                                                    <tr className="bg-white">
                                                                        <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-b border-slate-200 w-14 min-w-14">Alice</td>
                                                                        {fullPath.map((rec, i) => (
                                                                            <td key={i} className={["px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14", loopStartFull != null && i >= loopStartFull ? "bg-blue-50 text-blue-800" : ""].join(" ")}>
                                                                                {rec.a1Price.toFixed(1)}
                                                                            </td>
                                                                        ))}
                                                                    </tr>
                                                                    <tr className="bg-slate-50">
                                                                        <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-b border-slate-200 w-14 min-w-14">Bob</td>
                                                                        {fullPath.map((rec, i) => (
                                                                            <td key={i} className={["px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14", loopStartFull != null && i >= loopStartFull ? "bg-blue-50 text-blue-800" : ""].join(" ")}>
                                                                                {rec.a2Price.toFixed(1)}
                                                                            </td>
                                                                        ))}
                                                                    </tr>
                                                                    <tr className="bg-white">
                                                                        <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-slate-200 w-14 min-w-14 text-slate-400">Step</td>
                                                                        {fullPath.map((_, i) => (
                                                                            <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-400">
                                                                                {i}
                                                                            </td>
                                                                        ))}
                                                                    </tr>
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                        {loopStartFull != null && (
                                                            <p className="mt-1 text-xs text-blue-600">Blue cells = cycle</p>
                                                        )}
                                                        {fullPath.length > 10 && (
                                                            <p className="mt-1 text-xs text-slate-400">Scroll horizontally to see all steps.</p>
                                                        )}
                                                    </div>
                                                );
                                            })()}

                                            {trajSummary && (
                                                <div className="rounded-xl border p-3 space-y-1">
                                                    <div>Avg p1: {formatNum(trajSummary.avgP1, 3)}, Avg p2: {formatNum(trajSummary.avgP2, 3)}</div>
                                                    <div>Avg π1: {formatNum(trajSummary.avgPi1, 4)}, Avg π2: {formatNum(trajSummary.avgPi2, 4)}</div>
                                                    <div>Normalized π1: {trajSummary.norm1 == null ? "N/A" : formatNum(trajSummary.norm1, 3)}, Normalized π2: {trajSummary.norm2 == null ? "N/A" : formatNum(trajSummary.norm2, 3)}</div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        </div>

                        <Card className="rounded-2xl">
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <CardTitle>Custom Price Path Comparison</CardTitle>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => {
                                            setCustomPaths((prev) => [...prev, { id: nextCustomId, aliceInput: "", bobInput: "" }]);
                                            setNextCustomId((n) => n + 1);
                                        }}
                                    >
                                        + Add custom price path
                                    </Button>
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <p className="text-xs text-slate-500">
                                    Enter comma-separated prices for each player (e.g. <span className="font-mono">7, 8, 7, 8</span>). The cycle repeats. Profits are averaged over the entered sequence.
                                </p>
                                {customPaths.map((cp) => {
                                    const parseRow = (raw) =>
                                        raw.split(",").map((s) => parseFloat(s.trim())).filter((v) => Number.isFinite(v));
                                    const alicePrices = parseRow(cp.aliceInput);
                                    const bobPrices = parseRow(cp.bobInput);
                                    const len = Math.min(alicePrices.length, bobPrices.length);
                                    let avgPi1 = null, avgPi2 = null, avgP1 = null, avgP2 = null, norm1 = null, norm2 = null;
                                    if (len > 0) {
                                        let sumPi1 = 0, sumPi2 = 0, sumP1 = 0, sumP2 = 0;
                                        for (let i = 0; i < len; i++) {
                                            sumPi1 += profit1(alicePrices[i], bobPrices[i], appliedConfig.c, appliedConfig.k1, appliedConfig.k2);
                                            sumPi2 += profit2(alicePrices[i], bobPrices[i], appliedConfig.c, appliedConfig.k1, appliedConfig.k2);
                                            sumP1 += alicePrices[i];
                                            sumP2 += bobPrices[i];
                                        }
                                        avgPi1 = sumPi1 / len;
                                        avgPi2 = sumPi2 / len;
                                        avgP1 = sumP1 / len;
                                        avgP2 = sumP2 / len;
                                        const denom = pricing.piC - pricing.piE;
                                        if (Math.abs(denom) > 1e-10) {
                                            norm1 = (avgPi1 - pricing.piE) / denom;
                                            norm2 = (avgPi2 - pricing.piE) / denom;
                                        }
                                    }
                                    return (
                                        <div key={cp.id} className="rounded-xl border p-3 space-y-3">
                                            <div className="flex items-center justify-between">
                                                <span className="text-xs font-semibold text-slate-600">Path #{cp.id + 1}</span>
                                                {customPaths.length > 1 && (
                                                    <button
                                                        type="button"
                                                        className="text-xs text-slate-400 hover:text-red-500"
                                                        onClick={() => setCustomPaths((prev) => prev.filter((p) => p.id !== cp.id))}
                                                    >
                                                        Remove
                                                    </button>
                                                )}
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="space-y-1">
                                                    <Label className="text-xs">Alice prices (p1)</Label>
                                                    <Input
                                                        className="font-mono text-xs"
                                                        placeholder="e.g. 7, 8, 7, 8"
                                                        value={cp.aliceInput}
                                                        onChange={(e) =>
                                                            setCustomPaths((prev) =>
                                                                prev.map((p) => p.id === cp.id ? { ...p, aliceInput: e.target.value } : p)
                                                            )
                                                        }
                                                    />
                                                </div>
                                                <div className="space-y-1">
                                                    <Label className="text-xs">Bob prices (p2)</Label>
                                                    <Input
                                                        className="font-mono text-xs"
                                                        placeholder="e.g. 7, 8, 7, 8"
                                                        value={cp.bobInput}
                                                        onChange={(e) =>
                                                            setCustomPaths((prev) =>
                                                                prev.map((p) => p.id === cp.id ? { ...p, bobInput: e.target.value } : p)
                                                            )
                                                        }
                                                    />
                                                </div>
                                            </div>
                                            {len > 0 && (
                                                <>
                                                    <div className="overflow-x-auto rounded-xl border">
                                                        <table className="border-collapse text-xs" style={{ tableLayout: "fixed" }}>
                                                            <tbody>
                                                                <tr className="bg-white">
                                                                    <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-b border-slate-200 w-14 min-w-14">Alice</td>
                                                                    {alicePrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14">{p.toFixed(1)}</td>
                                                                    ))}
                                                                </tr>
                                                                <tr className="bg-slate-50">
                                                                    <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-b border-slate-200 w-14 min-w-14">Bob</td>
                                                                    {bobPrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14">{p.toFixed(1)}</td>
                                                                    ))}
                                                                </tr>
                                                                <tr className="bg-white">
                                                                    <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-slate-200 w-14 min-w-14 text-slate-400">π1</td>
                                                                    {alicePrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-600">
                                                                            {formatNum(profit1(p, bobPrices[i], appliedConfig.c, appliedConfig.k1, appliedConfig.k2), 2)}
                                                                        </td>
                                                                    ))}
                                                                </tr>
                                                                <tr className="bg-slate-50">
                                                                    <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-slate-200 w-14 min-w-14 text-slate-400">π2</td>
                                                                    {bobPrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-600">
                                                                            {formatNum(profit2(alicePrices[i], p, appliedConfig.c, appliedConfig.k1, appliedConfig.k2), 2)}
                                                                        </td>
                                                                    ))}
                                                                </tr>
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2 space-y-0.5">
                                                            <div className="font-semibold text-slate-700">Alice avg</div>
                                                            <div>p̄1 = {formatNum(avgP1, 3)}</div>
                                                            <div>π̄1 = {formatNum(avgPi1, 4)}</div>
                                                            {norm1 != null && <div className="text-slate-500">Δ1 = {formatNum(norm1, 3)}</div>}
                                                        </div>
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2 space-y-0.5">
                                                            <div className="font-semibold text-slate-700">Bob avg</div>
                                                            <div>p̄2 = {formatNum(avgP2, 3)}</div>
                                                            <div>π̄2 = {formatNum(avgPi2, 4)}</div>
                                                            {norm2 != null && <div className="text-slate-500">Δ2 = {formatNum(norm2, 3)}</div>}
                                                        </div>
                                                    </div>
                                                </>
                                            )}
                                            {len === 0 && cp.aliceInput && cp.bobInput && (
                                                <p className="text-xs text-amber-600">Enter at least one valid price in each row.</p>
                                            )}
                                        </div>
                                    );
                                })}
                            </CardContent>
                        </Card>

                        <div className="grid gap-6 xl:grid-cols-2">
                            <Card className="rounded-2xl">
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <CardTitle>Q1 Table (Alice)</CardTitle>
                                        <Button variant="outline" size="sm" onClick={() => exportQTableCSV(playbackFrame.Q1, env.states, env.prices, "q1_alice.csv")}>
                                            Export CSV
                                        </Button>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    <div className="max-h-[420px] overflow-auto rounded-xl border">
                                        <table className="min-w-full text-xs">
                                            <thead className="bg-slate-100">
                                                <tr>
                                                    <th className="px-3 py-2 text-left">State</th>
                                                    {env.prices.map((p) => (
                                                        <th key={`a1-${p}`} className="px-3 py-2 text-left">p={p.toFixed(1)}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>{renderQRows(playbackFrame.Q1, env.states, env.prices)}</tbody>
                                        </table>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="rounded-2xl">
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <CardTitle>Q2 Table (Bob)</CardTitle>
                                        <Button variant="outline" size="sm" onClick={() => exportQTableCSV(playbackFrame.Q2, env.states, env.prices, "q2_bob.csv")}>
                                            Export CSV
                                        </Button>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    <div className="max-h-[420px] overflow-auto rounded-xl border">
                                        <table className="min-w-full text-xs">
                                            <thead className="bg-slate-100">
                                                <tr>
                                                    <th className="px-3 py-2 text-left">State</th>
                                                    {env.prices.map((p) => (
                                                        <th key={`a2-${p}`} className="px-3 py-2 text-left">p={p.toFixed(1)}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>{renderQRows(playbackFrame.Q2, env.states, env.prices)}</tbody>
                                        </table>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
