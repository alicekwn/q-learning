import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, RotateCcw, StepForward, FastForward, ChevronLeft, ChevronRight, Bone, Dot, ArrowLeft, ArrowRight } from "lucide-react";

const ACTIONS = [
  { key: 0, label: "Left", symbol: "←", delta: -1 },
  { key: 1, label: "Right", symbol: "→", delta: 1 },
];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function argmax(arr) {
  let bestIdx = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[bestIdx]) bestIdx = i;
  }
  return bestIdx;
}

function makeQTable(numStates) {
  return Array.from({ length: numStates }, () => [0, 0]);
}

function deepCopyQ(q) {
  return q.map((row) => [...row]);
}

function formatNum(x) {
  return Number.isFinite(x) ? x.toFixed(2) : "0.00";
}

function getRandomStart(startPos, endPos, goalPos) {
  const candidates = [];
  for (let s = startPos; s <= endPos; s++) {
    if (s !== goalPos) candidates.push(s);
  }
  return candidates[Math.floor(Math.random() * candidates.length)] ?? startPos;
}

function getEpisodeStartIndex({ startPos, endPos, goalPos, randomStartEachEpisode, fixedStartAbsolute }) {
  if (randomStartEachEpisode) {
    return getRandomStart(startPos, endPos, goalPos) - startPos;
  }

  const clampedStart = clamp(fixedStartAbsolute, startPos, endPos);
  if (clampedStart === goalPos) {
    if (goalPos > startPos) return goalPos - 1 - startPos;
    if (goalPos < endPos) return goalPos + 1 - startPos;
  }
  return clampedStart - startPos;
}

function chooseAction(stateIndex, table, epsilon) {
  const explore = Math.random() < epsilon;
  if (explore) {
    return { action: Math.random() < 0.5 ? 0 : 1, mode: "explore" };
  }
  return { action: argmax(table[stateIndex]), mode: "exploit" };
}

function environmentStep(stateIndex, action, startPos, endPos, goalPos, rewardValue) {
  const absoluteState = stateIndex + startPos;
  const delta = ACTIONS[action].delta;
  const nextAbsolute = clamp(absoluteState + delta, startPos, endPos);
  const nextIndex = nextAbsolute - startPos;
  const done = nextAbsolute === goalPos;
  const reward = done ? rewardValue : -1;
  return { nextIndex, reward, done };
}

function buildUpdate({ episode, stateIndex, nextIndex, action, mode, reward, done, oldQ, maxNextQ, target, alpha, gamma, newQ }) {
  return {
    episode,
    stateIndex,
    nextIndex,
    action,
    actionLabel: ACTIONS[action].label,
    actionSymbol: ACTIONS[action].symbol,
    mode,
    reward,
    done,
    oldQ,
    maxNextQ,
    target,
    alpha,
    gamma,
    newQ,
    equation: "Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]",
    numericEquation: `${formatNum(oldQ)} + ${formatNum(alpha)} × (${formatNum(reward)} + ${formatNum(gamma)} × ${formatNum(maxNextQ)} − ${formatNum(oldQ)}) = ${formatNum(newQ)}`,
  };
}

function simulateOneStep(snapshot, params) {
  if (snapshot.episodeDone) {
    const nextDog = getEpisodeStartIndex(params);
    const nextEpisode = snapshot.episode + 1;
    const resetEntry = {
      id: snapshot.history.length,
      eventType: "episode_reset",
      episode: nextEpisode,
      stepCount: 0,
      dogIndex: nextDog,
      pathSequence: [nextDog],
      qTable: deepCopyQ(snapshot.qTable),
      note: "New episode begins",
    };

    return {
      ...snapshot,
      episode: nextEpisode,
      dogIndex: nextDog,
      pathSequence: [nextDog],
      stepCount: 0,
      episodeDone: nextDog === params.goalPos - params.startPos,
      lastUpdate: null,
      history: [...snapshot.history, resetEntry],
    };
  }

  const s = snapshot.dogIndex;
  const { action, mode } = chooseAction(s, snapshot.qTable, params.epsilon);
  const { nextIndex, reward, done } = environmentStep(s, action, params.startPos, params.endPos, params.goalPos, params.rewardValue);

  const oldQ = snapshot.qTable[s][action];
  const maxNextQ = done ? 0 : Math.max(...snapshot.qTable[nextIndex]);
  const target = reward + params.gamma * maxNextQ;
  const newQ = oldQ + params.alpha * (target - oldQ);

  const nextQTable = deepCopyQ(snapshot.qTable);
  nextQTable[s][action] = newQ;

  const nextPathSequence = [...snapshot.pathSequence, nextIndex];

  const update = buildUpdate({
    episode: snapshot.episode,
    stateIndex: s,
    nextIndex,
    action,
    mode,
    reward,
    done,
    oldQ,
    maxNextQ,
    target,
    alpha: params.alpha,
    gamma: params.gamma,
    newQ,
  });

  const nextStepCount = snapshot.stepCount + 1;
  const historyEntry = {
    id: snapshot.history.length,
    eventType: done ? "episode_end" : "step",
    episode: snapshot.episode,
    stepCount: nextStepCount,
    dogIndex: nextIndex,
    pathSequence: nextPathSequence,
    qTable: deepCopyQ(nextQTable),
    update,
    note: done ? "The dog found the bone!" : `${mode === "explore" ? "Exploration" : "Exploitation"}: ${ACTIONS[action].label}`,
  };

  return {
    ...snapshot,
    qTable: nextQTable,
    dogIndex: nextIndex,
    pathSequence: nextPathSequence,
    stepCount: nextStepCount,
    episodeDone: done,
    lastUpdate: update,
    history: [...snapshot.history, historyEntry],
  };
}

function simulateBatch(snapshot, params, episodesToRun) {
  let nextSnapshot = {
    ...snapshot,
    qTable: deepCopyQ(snapshot.qTable),
    pathSequence: [...snapshot.pathSequence],
    history: [...snapshot.history],
  };

  let completedEpisodes = 0;
  let guard = 0;
  const maxIterations = Math.max(episodesToRun * 500, 500);

  while (completedEpisodes < episodesToRun && guard < maxIterations) {
    guard += 1;
    const wasDone = nextSnapshot.episodeDone;
    nextSnapshot = simulateOneStep(nextSnapshot, params);

    if (!wasDone && nextSnapshot.episodeDone) {
      completedEpisodes += 1;
      if (completedEpisodes < episodesToRun) {
        nextSnapshot = simulateOneStep(nextSnapshot, params);
      }
    }
  }

  return nextSnapshot;
}

function GridCell({ index, dogPos, goalPos, isPlayback }) {
  const isDog = dogPos === index;
  const isGoal = goalPos === index;

  return (
    <div
      className={[
        "relative flex h-16 min-w-0 flex-1 items-center justify-center rounded-2xl border text-2xl shadow-sm",
        isGoal ? "border-amber-400 bg-amber-50" : "border-slate-200 bg-white",
        isPlayback ? "transition-all duration-200" : "",
      ].join(" ")}
    >
      <div className="absolute left-2 top-2 text-xs text-slate-400">{index}</div>
      <div className="flex items-center gap-1">
        {isDog && <span title="Dog">🐶</span>}
        {isGoal && <span title="Bone">🦴</span>}
      </div>
    </div>
  );
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

export default function QLearningDogBoneDemo() {
  const [startPos, setStartPos] = useState(0);
  const [endPos, setEndPos] = useState(6);
  const [goalPos, setGoalPos] = useState(5);
  const [fixedStartAbsolute, setFixedStartAbsolute] = useState(0);
  const [alpha, setAlpha] = useState(0.5);
  const [gamma, setGamma] = useState(0.9);
  const [epsilon, setEpsilon] = useState(0.25);
  const [rewardValue, setRewardValue] = useState(10);
  const [randomStartEachEpisode, setRandomStartEachEpisode] = useState(true);
  const [autoplay, setAutoplay] = useState(false);
  const [appliedConfig, setAppliedConfig] = useState({
    startPos: 0,
    endPos: 6,
    goalPos: 5,
    fixedStartAbsolute: 0,
    alpha: 0.5,
    gamma: 0.9,
    epsilon: 0.25,
    rewardValue: 10,
    randomStartEachEpisode: true,
  });

  const numStates = appliedConfig.endPos - appliedConfig.startPos + 1;
  const goalIndex = clamp(appliedConfig.goalPos - appliedConfig.startPos, 0, numStates - 1);

  const [qTable, setQTable] = useState(() => makeQTable(7));
  const [episode, setEpisode] = useState(1);
  const [stepCount, setStepCount] = useState(0);
  const [dogIndex, setDogIndex] = useState(0);
  const [episodeDone, setEpisodeDone] = useState(false);
  const [pathSequence, setPathSequence] = useState([0]);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [history, setHistory] = useState([]);
  const [playbackIndex, setPlaybackIndex] = useState(null);
  const autoplayRef = useRef(null);

  const params = useMemo(() => appliedConfig, [appliedConfig]);

  const hasPendingConfig = useMemo(() => {
    return (
      startPos !== appliedConfig.startPos ||
      endPos !== appliedConfig.endPos ||
      goalPos !== appliedConfig.goalPos ||
      fixedStartAbsolute !== appliedConfig.fixedStartAbsolute ||
      alpha !== appliedConfig.alpha ||
      gamma !== appliedConfig.gamma ||
      epsilon !== appliedConfig.epsilon ||
      rewardValue !== appliedConfig.rewardValue ||
      randomStartEachEpisode !== appliedConfig.randomStartEachEpisode
    );
  }, [startPos, endPos, goalPos, fixedStartAbsolute, alpha, gamma, epsilon, rewardValue, randomStartEachEpisode, appliedConfig]);

  const currentDisplay = playbackIndex == null ? null : history[playbackIndex];

  const playbackFrame = currentDisplay ?? {
    dogIndex,
    pathSequence,
    qTable,
    update: lastUpdate,
    note: "Live state",
    episode,
  };

  const displayedVisitedPath = useMemo(() => {
    const sequence = playbackFrame.pathSequence ?? [];
    if (sequence.length === 0) return "—";
    return sequence.map((idx) => idx + appliedConfig.startPos).join(" → ");
  }, [playbackFrame, appliedConfig.startPos]);

  const greedyPolicyDirections = useMemo(() => {
    const displayQTable = playbackFrame.qTable ?? qTable;
    return displayQTable.map((row, idx) => {
      const absoluteState = idx + appliedConfig.startPos;
      if (absoluteState === appliedConfig.goalPos) return "goal";
      if (row[0] === 0 && row[1] === 0) return "none";
      const best = argmax(row);
      return best === 0 ? "left" : "right";
    });
  }, [playbackFrame, qTable, appliedConfig.startPos, appliedConfig.goalPos]);

  const stats = useMemo(() => {
    const finishedEpisodes = history.filter((h) => h.eventType === "episode_end");
    const wins = finishedEpisodes.length;
    const avgSteps = wins > 0 ? finishedEpisodes.reduce((a, h) => a + h.stepCount, 0) / wins : 0;
    return { finishedEpisodes: wins, avgSteps };
  }, [history]);

  function applySnapshot(snapshot, options = {}) {
    setQTable(snapshot.qTable);
    setEpisode(snapshot.episode);
    setStepCount(snapshot.stepCount);
    setDogIndex(snapshot.dogIndex);
    setEpisodeDone(snapshot.episodeDone);
    setPathSequence(snapshot.pathSequence);
    setLastUpdate(snapshot.lastUpdate);
    setHistory(snapshot.history);
    if (options.resetPlayback) {
      setPlaybackIndex(null);
    }
  }

  function buildCurrentSnapshot(overrides = {}) {
    return {
      qTable,
      episode,
      stepCount,
      dogIndex,
      episodeDone,
      pathSequence,
      lastUpdate,
      history,
      ...overrides,
    };
  }

  function buildFreshSnapshot(customParams = params) {
    const newNumStates = customParams.endPos - customParams.startPos + 1;
    const freshQ = makeQTable(newNumStates);
    const initialDog = getEpisodeStartIndex(customParams);
    return {
      qTable: freshQ,
      episode: 1,
      stepCount: 0,
      dogIndex: initialDog,
      episodeDone: initialDog === customParams.goalPos - customParams.startPos,
      pathSequence: [initialDog],
      lastUpdate: null,
      history: [],
    };
  }

  useEffect(() => {
    const newGoal = clamp(goalPos, startPos, endPos);
    if (newGoal !== goalPos) setGoalPos(newGoal);

    const newFixedStart = clamp(fixedStartAbsolute, startPos, endPos);
    if (newFixedStart !== fixedStartAbsolute) setFixedStartAbsolute(newFixedStart);
  }, [startPos, endPos, goalPos, fixedStartAbsolute]);


  useEffect(() => {
    if (!autoplay) {
      if (autoplayRef.current) clearInterval(autoplayRef.current);
      autoplayRef.current = null;
      return;
    }

    autoplayRef.current = setInterval(() => {
      setPlaybackIndex(null);
      const nextSnapshot = simulateOneStep(buildCurrentSnapshot(), params);
      applySnapshot(nextSnapshot);
    }, 500);

    return () => {
      if (autoplayRef.current) clearInterval(autoplayRef.current);
    };
  }, [autoplay, qTable, episode, stepCount, dogIndex, episodeDone, pathSequence, lastUpdate, history, params]);

  function configureEnvironment() {
    const nextConfig = {
      startPos,
      endPos,
      goalPos: clamp(goalPos, startPos, endPos),
      fixedStartAbsolute: clamp(fixedStartAbsolute, startPos, endPos),
      alpha,
      gamma,
      epsilon,
      rewardValue,
      randomStartEachEpisode,
    };

    setAppliedConfig(nextConfig);
    applySnapshot(buildFreshSnapshot(nextConfig), { resetPlayback: true });
    setAutoplay(false);
  }

  function resetEpisode() {
    const nextDog = getEpisodeStartIndex(params);
    const snapshot = buildCurrentSnapshot({
      dogIndex: nextDog,
      pathSequence: [nextDog],
      stepCount: 0,
      episodeDone: nextDog === goalIndex,
      lastUpdate: null,
    });
    applySnapshot(snapshot, { resetPlayback: true });
  }

  function resetAll() {
    applySnapshot(buildFreshSnapshot(params), { resetPlayback: true });
    setAutoplay(false);
  }

  function stepEpisode() {
    const nextSnapshot = simulateOneStep(buildCurrentSnapshot(), params);
    applySnapshot(nextSnapshot, { resetPlayback: true });
  }

  function trainBatch(episodesToRun) {
    const nextSnapshot = simulateBatch(buildCurrentSnapshot(), params, episodesToRun);
    applySnapshot(nextSnapshot, { resetPlayback: true });
  }

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Q-Learning: Teach the Dog to Find the Bone</h1>
          <p className="max-w-4xl text-slate-600">
            This demo shows how a dog learns on a 1D grid by trial and error. Use the controls to explore
            <span className="font-semibold"> exploration vs exploitation</span>, the <span className="font-semibold">learning rate α</span>, and the
            <span className="font-semibold"> discount factor γ</span>.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-[360px_1fr]">
          <div className="space-y-6">
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle>Algorithm Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="grid grid-cols-3 gap-3">
                  <div className="space-y-2">
                    <Label>Grid Start</Label>
                    <Input
                      type="number"
                      value={startPos}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isNaN(v) && v < endPos) setStartPos(v);
                      }}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Grid End</Label>
                    <Input
                      type="number"
                      value={endPos}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isNaN(v) && v > startPos) setEndPos(v);
                      }}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Goal Position</Label>
                    <Input
                      type="number"
                      value={goalPos}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isNaN(v)) setGoalPos(clamp(v, startPos, endPos));
                      }}
                    />
                  </div>
                </div>

                {!randomStartEachEpisode && (
                  <div className="space-y-2">
                    <Label>Fixed Start Position</Label>
                    <Input
                      type="number"
                      value={fixedStartAbsolute}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isNaN(v)) {
                          setFixedStartAbsolute(clamp(v, startPos, endPos));
                        }
                      }}
                    />
                  </div>
                )}

                <ParamSlider label="Learning Rate (Alpha)" min={0} max={1} step={0.01} value={alpha} onChange={setAlpha} formatValue={(v) => v.toFixed(2)} />
                <ParamSlider label="Discount Factor (Gamma)" min={0} max={0.99} step={0.01} value={gamma} onChange={setGamma} formatValue={(v) => v.toFixed(2)} />
                <ParamSlider label="Exploration Rate (Epsilon)" min={0} max={1} step={0.01} value={epsilon} onChange={setEpsilon} formatValue={(v) => v.toFixed(2)} />
                <ParamSlider label="Goal Reward" min={1} max={30} step={1} value={rewardValue} onChange={setRewardValue} />

                <div className="flex items-center justify-between rounded-xl border p-3">
                  <div>
                    <Label>Random start each episode</Label>
                    <p className="text-xs text-slate-500">Turn on to sample a new non-goal starting state every episode.</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setRandomStartEachEpisode((v) => !v)}
                    className={[
                      "inline-flex h-6 w-11 items-center rounded-full transition",
                      randomStartEachEpisode ? "bg-slate-900" : "bg-slate-300",
                    ].join(" ")}
                    aria-pressed={randomStartEachEpisode}
                  >
                    <span
                      className={[
                        "inline-block h-5 w-5 rounded-full bg-white transition",
                        randomStartEachEpisode ? "translate-x-5" : "translate-x-0.5",
                      ].join(" ")}
                    />
                  </button>
                </div>
                <Button onClick={configureEnvironment} disabled={!hasPendingConfig} className="w-full">
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
                  <Button variant="outline" onClick={resetEpisode}><RotateCcw className="mr-2 h-4 w-4" />Reset Episode</Button>
                  <Button variant="outline" onClick={resetAll}>Full Reset</Button>
                  <Button onClick={stepEpisode}><StepForward className="mr-2 h-4 w-4" />Step</Button>
                  <Button variant="secondary" onClick={() => trainBatch(10)}><FastForward className="mr-2 h-4 w-4" />Train 10</Button>
                  <Button variant="secondary" onClick={() => trainBatch(100)}>Train 100</Button>
                  <Button variant={autoplay ? "destructive" : "default"} onClick={() => setAutoplay((v) => !v)}>
                    {autoplay ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                    {autoplay ? "Stop" : "Autoplay"}
                  </Button>
                </div>

                {hasPendingConfig && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                    Parameters changed. Click <span className="font-semibold">Configure</span> to restart training with the new environment.
                  </div>
                )}

                <div className="rounded-xl border p-3">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <div className="font-medium">Playback Controls</div>
                      <div className="text-xs text-slate-500">Browse earlier training states without a separate playback card.</div>
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
                    <div className="text-xs text-slate-500">
                      {history.length === 0 ? "No history yet." : `${history.length} saved states`}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card className="rounded-2xl">
              <CardHeader>
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <CardTitle>1D Grid World</CardTitle>
                  <div className="flex flex-wrap items-center gap-2 text-sm">
                    <Badge variant="secondary">Live episode: {episode}</Badge>
                    <Badge>{playbackFrame.note}</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="rounded-xl border p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="font-semibold">Visited path in displayed run</h3>
                    <span className="text-xs text-slate-500">Chronological state sequence</span>
                  </div>
                  <div className="rounded-lg bg-slate-100 px-3 py-2 font-mono text-sm text-slate-700">
                    {displayedVisitedPath}
                  </div>
                </div>

                <div className="grid gap-2 md:gap-3" style={{ gridTemplateColumns: `repeat(${numStates}, minmax(0, 1fr))` }}>
                  {Array.from({ length: numStates }, (_, idx) => (
                    <GridCell
                      key={idx}
                      index={idx + appliedConfig.startPos}
                      dogPos={playbackFrame.dogIndex + appliedConfig.startPos}
                      goalPos={appliedConfig.goalPos}
                      isPlayback={playbackIndex != null}
                    />
                  ))}
                </div>

                <div className="rounded-xl border p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="font-semibold">Greedy policy implied by current Q-table</h3>
                    <span className="text-xs text-slate-500">Arrow at each state shows the current best action</span>
                  </div>
                  <div className="mb-3 text-sm text-slate-600">
                    Read from the Q-matrix: arrows show the optimal action at each position. A dot means the state has not learned a preference yet.
                  </div>
                  <div className="grid gap-2 md:gap-3" style={{ gridTemplateColumns: `repeat(${numStates}, minmax(0, 1fr))` }}>
                    {greedyPolicyDirections.map((symbol, idx) => {
                      const absoluteState = idx + appliedConfig.startPos
                      const isGoal = absoluteState === appliedConfig.goalPos;
                      return (
                        <div
                          key={`policy-${absoluteState}`}
                          className={[
                            "flex h-20 min-w-0 flex-col items-center justify-center rounded-xl border shadow-sm",
                            isGoal ? "border-amber-400 bg-amber-100" : "border-sky-200 bg-sky-100",
                          ].join(" ")}
                        >
                          <div className="flex items-center justify-center">
                            {symbol === "goal" && <Bone className="h-9 w-9 text-amber-700" />}
                            {symbol === "none" && <Dot className="h-9 w-9 text-slate-500" />}
                            {symbol === "left" && <ArrowLeft className="h-9 w-9 text-indigo-700" />}
                            {symbol === "right" && <ArrowRight className="h-9 w-9 text-indigo-700" />}
                          </div>
                          <div className="mt-2 text-sm text-slate-700">{absoluteState}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
            <div className="grid gap-6 md:grid-cols-2">
              <Card className="rounded-2xl">
                <CardHeader>
                  <CardTitle>Bellman Equation Update</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {!playbackFrame.update && <p className="text-sm text-slate-500">No update yet. Press Step to perform one Q-learning update.</p>}
                  {playbackFrame.update && (
                    <>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="rounded-xl border p-3"><span className="text-slate-500">State</span><div className="text-lg font-semibold">{playbackFrame.update.stateIndex + appliedConfig.startPos}</div></div>
                        <div className="rounded-xl border p-3"><span className="text-slate-500">Next state</span><div className="text-lg font-semibold">{playbackFrame.update.nextIndex + appliedConfig.startPos}</div></div>
                        <div className="rounded-xl border p-3"><span className="text-slate-500">Action</span><div className="text-lg font-semibold">{playbackFrame.update.actionSymbol} {playbackFrame.update.actionLabel}</div></div>
                        <div className="rounded-xl border p-3"><span className="text-slate-500">Mode</span><div className="text-lg font-semibold">{playbackFrame.update.mode}</div></div>
                        <div className="rounded-xl border p-3"><span className="text-slate-500">Reward</span><div className="text-lg font-semibold">{playbackFrame.update.reward}</div></div>
                        <div className="rounded-xl border p-3"><span className="text-slate-500">Done?</span><div className="text-lg font-semibold">{playbackFrame.update.done ? "Yes" : "No"}</div></div>
                      </div>

                      <div className="rounded-xl bg-slate-900 p-4 font-mono text-sm text-slate-100">
                        <div>{playbackFrame.update.equation}</div>
                        <div className="mt-3 text-sky-300">{playbackFrame.update.numericEquation}</div>
                      </div>

                      <div className="space-y-2 text-sm text-slate-700">
                        <p><span className="font-semibold">Old Q:</span> {formatNum(playbackFrame.update.oldQ)}</p>
                        <p><span className="font-semibold">Target:</span> reward + γ × best future value = {formatNum(playbackFrame.update.target)}</p>
                        <p><span className="font-semibold">New Q:</span> {formatNum(playbackFrame.update.newQ)}</p>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              <Card className="rounded-2xl">
                <CardHeader>
                  <CardTitle>Q-Matrix Table</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-auto rounded-xl border">
                    <table className="min-w-full text-sm">
                      <thead className="bg-slate-100">
                        <tr>
                          <th className="px-4 py-3 text-left">State</th>
                          <th className="px-4 py-3 text-left">Left</th>
                          <th className="px-4 py-3 text-left">Right</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(playbackFrame.qTable ?? qTable).map((row, idx) => {
                          const absoluteState = idx + appliedConfig.startPos
                          const isUnlearned = row[0] === 0 && row[1] === 0;
                          const best = argmax(row);
                          return (
                            <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                              <td className="px-4 py-3 font-medium">{absoluteState}{idx === goalIndex ? " 🦴" : ""}</td>
                              <td className={["px-4 py-3 font-mono", !isUnlearned && best === 0 ? "bg-emerald-50 text-emerald-700" : ""].join(" ")}>{formatNum(row[0])}</td>
                              <td className={["px-4 py-3 font-mono", !isUnlearned && best === 1 ? "bg-emerald-50 text-emerald-700" : ""].join(" ")}>{formatNum(row[1])}</td>
                            </tr>
                          );
                        })}
                      </tbody>
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
