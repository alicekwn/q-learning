import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, RotateCcw, StepForward, FastForward, ChevronLeft, ChevronRight, Bone, Dot, ArrowLeft, ArrowRight, ArrowUp, ArrowDown, ArrowUpRight, ArrowUpLeft, ArrowDownRight, ArrowDownLeft } from "lucide-react";

const ACTIONS = [
    { key: 0, label: "Up", short: "U", dx: 0, dy: 1 },
    { key: 1, label: "Down", short: "D", dx: 0, dy: -1 },
    { key: 2, label: "Left", short: "L", dx: -1, dy: 0 },
    { key: 3, label: "Right", short: "R", dx: 1, dy: 0 },
];

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}
function formatNum(x) {
    return Number.isFinite(x) ? x.toFixed(2) : "0.00";
}

function makeStateKey(x, y) {
    return `${x},${y}`;
}

function parseStateKey(key) {
    const [x, y] = key.split(",").map(Number);
    return { x, y };
}

function enumerateStates(config) {
    const states = [];
    for (let y = config.yMax; y >= config.yMin; y -= 1) {
        for (let x = config.xMin; x <= config.xMax; x += 1) {
            states.push({ x, y, key: makeStateKey(x, y) });
        }
    }
    return states;
}

function buildStateMaps(config) {
    const states = enumerateStates(config);
    const indexByKey = new Map();
    states.forEach((state, idx) => indexByKey.set(state.key, idx));
    return { states, indexByKey };
}

function makeQTable(states) {
    return states.map((state) => ({ key: state.key, values: [0, 0, 0, 0] }));
}

function deepCopyQTable(qTable) {
    return qTable.map((row) => ({ key: row.key, values: [...row.values] }));
}

function getRandomNonGoalState(config) {
    const candidates = [];
    for (let y = config.yMin; y <= config.yMax; y += 1) {
        for (let x = config.xMin; x <= config.xMax; x += 1) {
            if (!(x === config.goalX && y === config.goalY)) {
                candidates.push({ x, y });
            }
        }
    }
    return candidates[Math.floor(Math.random() * candidates.length)] ?? { x: config.xMin, y: config.yMin };
}

function normalizeFixedStart(config) {
    const x = clamp(config.fixedStartX, config.xMin, config.xMax);
    const y = clamp(config.fixedStartY, config.yMin, config.yMax);
    if (x === config.goalX && y === config.goalY) {
        if (x > config.xMin) return { x: x - 1, y };
        if (x < config.xMax) return { x: x + 1, y };
        if (y > config.yMin) return { x, y: y - 1 };
        if (y < config.yMax) return { x, y: y + 1 };
    }
    return { x, y };
}

function getEpisodeStartState(config) {
    if (config.randomStartEachEpisode) return getRandomNonGoalState(config);
    return normalizeFixedStart(config);
}

function getBestActionIndices(values) {
    const maxValue = Math.max(...values);
    return values.reduce((acc, value, idx) => {
        if (value === maxValue) acc.push(idx);
        return acc;
    }, []);
}

function chooseAction(stateIndex, qTable, epsilon) {
    const values = qTable[stateIndex].values;
    const explore = Math.random() < epsilon;
    if (explore) {
        const randomAction = Math.floor(Math.random() * ACTIONS.length);
        return { action: randomAction, mode: "explore", tiedBestActions: getBestActionIndices(values) };
    }
    const bestActions = getBestActionIndices(values);
    const greedyAction = bestActions[Math.floor(Math.random() * bestActions.length)];
    return { action: greedyAction, mode: "greedy", tiedBestActions: bestActions };
}

function environmentStep(state, actionIndex, config) {
    const action = ACTIONS[actionIndex];
    const proposedX = state.x + action.dx;
    const proposedY = state.y + action.dy;
    const nextX = clamp(proposedX, config.xMin, config.xMax);
    const nextY = clamp(proposedY, config.yMin, config.yMax);
    const done = nextX === config.goalX && nextY === config.goalY;
    const reward = done ? config.rewardValue : 0;
    return { nextState: { x: nextX, y: nextY }, reward, done };
}

function getPolicySymbol(values, isGoal) {
    if (isGoal) return "goal";
    const maxValue = Math.max(...values);
    if (maxValue === 0 && values.every((v) => v === 0)) return "none";
    const best = getBestActionIndices(values);
    if (best.length === 1) {
        if (best[0] === 0) return "up";
        if (best[0] === 1) return "down";
        if (best[0] === 2) return "left";
        if (best[0] === 3) return "right";
    }
    const key = [...best].sort((a, b) => a - b).join(",");
    if (key === "0,3") return "upRight";
    if (key === "0,2") return "upLeft";
    if (key === "1,3") return "downRight";
    if (key === "1,2") return "downLeft";
    return "none";
}

function buildUpdateLog({ episode, stepInEpisode, state, actionIndex, nextState, reward, done, mode, oldQ, maxNextQ, bestNextActions, alpha, gamma, target, newQ }) {
    const action = ACTIONS[actionIndex];
    return {
        episode,
        stepInEpisode,
        state,
        actionIndex,
        actionLabel: action.label,
        actionShort: action.short,
        nextState,
        reward,
        done,
        mode,
        oldQ,
        maxNextQ,
        bestNextActions,
        alpha,
        gamma,
        target,
        newQ,
        equation: "Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]",
        numericEquation: `${formatNum(oldQ)} + ${formatNum(alpha)} × (${formatNum(reward)} + ${formatNum(gamma)} × ${formatNum(maxNextQ)} − ${formatNum(oldQ)}) = ${formatNum(newQ)}`,
    };
}

function buildFreshSnapshot(config, stateMaps) {
    const startState = getEpisodeStartState(config);
    const startIndex = stateMaps.indexByKey.get(makeStateKey(startState.x, startState.y));
    return {
        qTable: makeQTable(stateMaps.states),
        completedEpisodes: 0,
        currentEpisodeNumber: 1,
        stepCount: 0,
        dogState: startState,
        dogIndex: startIndex,
        episodeDone: startState.x === config.goalX && startState.y === config.goalY,
        pathSequence: [startState],
        lastUpdate: null,
        history: [],
        stepsPerEpisode: [],
        qSnapshots: [],
    };
}

function simulateOneStep(snapshot, config, stateMaps) {
    if (snapshot.episodeDone) {
        const nextStartState = getEpisodeStartState(config);
        const nextStartIndex = stateMaps.indexByKey.get(makeStateKey(nextStartState.x, nextStartState.y));
        const resetEntry = {
            id: snapshot.history.length,
            eventType: "episode_reset",
            episode: snapshot.currentEpisodeNumber + 1,
            dogState: nextStartState,
            dogIndex: nextStartIndex,
            pathSequence: [nextStartState],
            qTable: deepCopyQTable(snapshot.qTable),
            note: "New episode begins",
        };
        return {
            ...snapshot,
            currentEpisodeNumber: snapshot.currentEpisodeNumber + 1,
            stepCount: 0,
            dogState: nextStartState,
            dogIndex: nextStartIndex,
            episodeDone: nextStartState.x === config.goalX && nextStartState.y === config.goalY,
            pathSequence: [nextStartState],
            lastUpdate: null,
            history: [...snapshot.history, resetEntry],
        };
    }

    const state = snapshot.dogState;
    const stateIndex = snapshot.dogIndex;
    const { action, mode } = chooseAction(stateIndex, snapshot.qTable, config.epsilon);
    const { nextState, reward, done } = environmentStep(state, action, config);
    const nextKey = makeStateKey(nextState.x, nextState.y);
    const nextIndex = stateMaps.indexByKey.get(nextKey);
    const oldQ = snapshot.qTable[stateIndex].values[action];
    const nextValues = snapshot.qTable[nextIndex].values;
    const maxNextQ = done ? 0 : Math.max(...nextValues);
    const bestNextActions = done ? [] : getBestActionIndices(nextValues).map((idx) => ACTIONS[idx].short);
    const target = reward + config.gamma * maxNextQ;
    const newQ = oldQ + config.alpha * (target - oldQ);

    const nextQTable = deepCopyQTable(snapshot.qTable);
    nextQTable[stateIndex].values[action] = newQ;

    const nextPathSequence = [...snapshot.pathSequence, nextState];
    const stepInEpisode = snapshot.stepCount + 1;
    const update = buildUpdateLog({
        episode: snapshot.currentEpisodeNumber,
        stepInEpisode,
        state,
        actionIndex: action,
        nextState,
        reward,
        done,
        mode,
        oldQ,
        maxNextQ,
        bestNextActions,
        alpha: config.alpha,
        gamma: config.gamma,
        target,
        newQ,
    });

    const historyEntry = {
        id: snapshot.history.length,
        eventType: done ? "episode_end" : "step",
        episode: snapshot.currentEpisodeNumber,
        stepInEpisode,
        dogState: nextState,
        dogIndex: nextIndex,
        pathSequence: nextPathSequence,
        qTable: deepCopyQTable(nextQTable),
        update,
        note: done ? "The dog reached the bone!" : `${mode === "explore" ? "Exploration" : "Greedy choice"}: ${ACTIONS[action].label}`,
    };

    return {
        ...snapshot,
        qTable: nextQTable,
        dogState: nextState,
        dogIndex: nextIndex,
        stepCount: stepInEpisode,
        episodeDone: done,
        pathSequence: nextPathSequence,
        lastUpdate: update,
        history: [...snapshot.history, historyEntry],
        completedEpisodes: done ? snapshot.completedEpisodes + 1 : snapshot.completedEpisodes,
        stepsPerEpisode: done ? [...snapshot.stepsPerEpisode, stepInEpisode] : snapshot.stepsPerEpisode,
        qSnapshots: done
            ? [...snapshot.qSnapshots, { episode: snapshot.completedEpisodes + 1, qTable: deepCopyQTable(nextQTable) }]
            : snapshot.qSnapshots,
    };
}

function simulateBatch(snapshot, config, stateMaps, episodesToRun) {
    let nextSnapshot = {
        ...snapshot,
        qTable: deepCopyQTable(snapshot.qTable),
        pathSequence: [...snapshot.pathSequence],
        history: [...snapshot.history],
        stepsPerEpisode: [...snapshot.stepsPerEpisode],
        qSnapshots: [...snapshot.qSnapshots],
    };

    let completed = 0;
    let guard = 0;
    const maxIterations = Math.max(episodesToRun * 1500, 1500);

    while (completed < episodesToRun && guard < maxIterations) {
        guard += 1;
        const wasDone = nextSnapshot.episodeDone;
        nextSnapshot = simulateOneStep(nextSnapshot, config, stateMaps);
        if (!wasDone && nextSnapshot.episodeDone) {
            completed += 1;
            if (completed < episodesToRun) {
                nextSnapshot = simulateOneStep(nextSnapshot, config, stateMaps);
            }
        }
    }
    return nextSnapshot;
}

function runSanityChecks() {
    const checks = [];
    checks.push({
        name: "clamp respects numeric bounds",
        pass: clamp(10, 0, 4) === 4 && clamp(-2, 0, 4) === 0 && clamp(3, 0, 4) === 3,
    });

    const config = {
        xMin: 0,
        xMax: 2,
        yMin: 0,
        yMax: 2,
        goalX: 2,
        goalY: 2,
        fixedStartX: 0,
        fixedStartY: 0,
        alpha: 0.5,
        gamma: 0.9,
        epsilon: 0,
        rewardValue: 10,
        randomStartEachEpisode: false,
    };
    const maps = buildStateMaps(config);
    checks.push({
        name: "grid enumerates every cell",
        pass: maps.states.length === 9,
    });

    const wallStep = environmentStep({ x: 0, y: 2 }, 0, config);
    checks.push({
        name: "moving beyond top wall stays in place",
        pass: wallStep.nextState.x === 0 && wallStep.nextState.y === 2,
    });

    const goalStep = environmentStep({ x: 1, y: 2 }, 3, config);
    checks.push({
        name: "goal reward appears only on landing on goal",
        pass: goalStep.done === true && goalStep.reward === 10,
    });

    const diagPolicy = getPolicySymbol([1, 0, 0, 1], false);
    checks.push({
        name: "diagonal policy summary detects up-right tie",
        pass: diagPolicy === "upRight",
    });

    const snapshot = buildFreshSnapshot(config, maps);
    checks.push({
        name: "fresh snapshot path starts with one state",
        pass: snapshot.pathSequence.length === 1,
    });

    return checks;
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

function PolicyIcon({ symbol }) {
    if (symbol === "goal") return <Bone className="h-7 w-7 text-amber-700" />;
    if (symbol === "up") return <ArrowUp className="h-7 w-7 text-indigo-700" />;
    if (symbol === "down") return <ArrowDown className="h-7 w-7 text-indigo-700" />;
    if (symbol === "left") return <ArrowLeft className="h-7 w-7 text-indigo-700" />;
    if (symbol === "right") return <ArrowRight className="h-7 w-7 text-indigo-700" />;
    if (symbol === "upRight") return <ArrowUpRight className="h-7 w-7 text-indigo-700" />;
    if (symbol === "upLeft") return <ArrowUpLeft className="h-7 w-7 text-indigo-700" />;
    if (symbol === "downRight") return <ArrowDownRight className="h-7 w-7 text-indigo-700" />;
    if (symbol === "downLeft") return <ArrowDownLeft className="h-7 w-7 text-indigo-700" />;
    return <Dot className="h-7 w-7 text-slate-500" />;
}

function GridCell({ state, dogState, goalState, isPlayback }) {
    const isDog = dogState.x === state.x && dogState.y === state.y;
    const isGoal = goalState.x === state.x && goalState.y === state.y;
    return (
        <div
            className={[
                "relative flex aspect-square min-w-0 items-center justify-center rounded-2xl border shadow-sm",
                isGoal ? "border-amber-400 bg-amber-50" : "border-slate-200 bg-white",
                isPlayback ? "transition-all duration-200" : "",
            ].join(" ")}
        >
            <div className="absolute left-2 top-2 text-[10px] text-slate-400">({state.x},{state.y})</div>
            <div className="flex items-center gap-1 text-2xl">
                {isDog && <span>🐶</span>}
                {isGoal && <span>🥩</span>}
            </div>
        </div>
    );
}

export default function QLearningDogBoneDemo2D() {
    const [xMin, setXMin] = useState(0);
    const [xMax, setXMax] = useState(3);
    const [yMin, setYMin] = useState(0);
    const [yMax, setYMax] = useState(3);
    const [goalX, setGoalX] = useState(3);
    const [goalY, setGoalY] = useState(3);
    const [fixedStartX, setFixedStartX] = useState(0);
    const [fixedStartY, setFixedStartY] = useState(0);
    const [alpha, setAlpha] = useState(0.5);
    const [gamma, setGamma] = useState(0.9);
    const [epsilon, setEpsilon] = useState(0.2);
    const [rewardValue, setRewardValue] = useState(10);
    const [randomStartEachEpisode, setRandomStartEachEpisode] = useState(false);
    const [autoplay, setAutoplay] = useState(false);
    const [appliedConfig, setAppliedConfig] = useState({
        xMin: 0,
        xMax: 3,
        yMin: 0,
        yMax: 3,
        goalX: 3,
        goalY: 3,
        fixedStartX: 0,
        fixedStartY: 0,
        alpha: 0.5,
        gamma: 0.9,
        epsilon: 0.2,
        rewardValue: 10,
        randomStartEachEpisode: false,
    });

    const autoplayRef = useRef(null);
    const stateMaps = useMemo(() => buildStateMaps(appliedConfig), [appliedConfig]);
    const sanityChecks = useMemo(() => runSanityChecks(), []);
    const params = useMemo(() => appliedConfig, [appliedConfig]);

    const [qTable, setQTable] = useState(() => makeQTable(stateMaps.states));
    const [completedEpisodes, setCompletedEpisodes] = useState(0);
    const [currentEpisodeNumber, setCurrentEpisodeNumber] = useState(1);
    const [stepCount, setStepCount] = useState(0);
    const [dogState, setDogState] = useState({ x: 0, y: 0 });
    const [dogIndex, setDogIndex] = useState(0);
    const [episodeDone, setEpisodeDone] = useState(false);
    const [pathSequence, setPathSequence] = useState([{ x: 0, y: 0 }]);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [history, setHistory] = useState([]);
    const [playbackIndex, setPlaybackIndex] = useState(null);
    const [stepsPerEpisode, setStepsPerEpisode] = useState([]);
    const [qSnapshots, setQSnapshots] = useState([]);

    const currentDisplay = playbackIndex == null ? null : history[playbackIndex];
    const playbackFrame = currentDisplay ?? {
        dogState,
        dogIndex,
        pathSequence,
        qTable,
        update: lastUpdate,
        note: "Live state",
        episode: currentEpisodeNumber,
    };

    const displayedPath = useMemo(() => {
        const sequence = playbackFrame.pathSequence ?? [];
        if (sequence.length === 0) return "—";
        return sequence.map((p) => `(${p.x},${p.y})`).join(" → ");
    }, [playbackFrame]);

    const policyRows = useMemo(() => {
        const table = playbackFrame.qTable ?? qTable;
        return stateMaps.states.map((state, idx) => ({
            ...state,
            index: idx,
            symbol: getPolicySymbol(table[idx].values, state.x === appliedConfig.goalX && state.y === appliedConfig.goalY),
        }));
    }, [playbackFrame, qTable, stateMaps.states, appliedConfig.goalX, appliedConfig.goalY]);

    const hasPendingConfig = useMemo(() => {
        return (
            xMin !== appliedConfig.xMin ||
            xMax !== appliedConfig.xMax ||
            yMin !== appliedConfig.yMin ||
            yMax !== appliedConfig.yMax ||
            goalX !== appliedConfig.goalX ||
            goalY !== appliedConfig.goalY ||
            fixedStartX !== appliedConfig.fixedStartX ||
            fixedStartY !== appliedConfig.fixedStartY ||
            alpha !== appliedConfig.alpha ||
            gamma !== appliedConfig.gamma ||
            epsilon !== appliedConfig.epsilon ||
            rewardValue !== appliedConfig.rewardValue ||
            randomStartEachEpisode !== appliedConfig.randomStartEachEpisode
        );
    }, [xMin, xMax, yMin, yMax, goalX, goalY, fixedStartX, fixedStartY, alpha, gamma, epsilon, rewardValue, randomStartEachEpisode, appliedConfig]);

    function applySnapshot(snapshot, options = {}) {
        setQTable(snapshot.qTable);
        setCompletedEpisodes(snapshot.completedEpisodes);
        setCurrentEpisodeNumber(snapshot.currentEpisodeNumber);
        setStepCount(snapshot.stepCount);
        setDogState(snapshot.dogState);
        setDogIndex(snapshot.dogIndex);
        setEpisodeDone(snapshot.episodeDone);
        setPathSequence(snapshot.pathSequence);
        setLastUpdate(snapshot.lastUpdate);
        setHistory(snapshot.history);
        setStepsPerEpisode(snapshot.stepsPerEpisode);
        setQSnapshots(snapshot.qSnapshots);
        if (options.resetPlayback) setPlaybackIndex(null);
    }

    function buildCurrentSnapshot(overrides = {}) {
        return {
            qTable,
            completedEpisodes,
            currentEpisodeNumber,
            stepCount,
            dogState,
            dogIndex,
            episodeDone,
            pathSequence,
            lastUpdate,
            history,
            stepsPerEpisode,
            qSnapshots,
            ...overrides,
        };
    }

    useEffect(() => {
        const nextGoalX = clamp(goalX, xMin, xMax);
        const nextGoalY = clamp(goalY, yMin, yMax);
        if (nextGoalX !== goalX) setGoalX(nextGoalX);
        if (nextGoalY !== goalY) setGoalY(nextGoalY);
        const nextStartX = clamp(fixedStartX, xMin, xMax);
        const nextStartY = clamp(fixedStartY, yMin, yMax);
        if (nextStartX !== fixedStartX) setFixedStartX(nextStartX);
        if (nextStartY !== fixedStartY) setFixedStartY(nextStartY);
    }, [xMin, xMax, yMin, yMax, goalX, goalY, fixedStartX, fixedStartY]);

    useEffect(() => {
        if (!autoplay) {
            if (autoplayRef.current) clearInterval(autoplayRef.current);
            autoplayRef.current = null;
            return;
        }
        autoplayRef.current = setInterval(() => {
            setPlaybackIndex(null);
            const nextSnapshot = simulateOneStep(buildCurrentSnapshot(), params, stateMaps);
            applySnapshot(nextSnapshot);
        }, 350);
        return () => {
            if (autoplayRef.current) clearInterval(autoplayRef.current);
        };
    }, [autoplay, qTable, completedEpisodes, currentEpisodeNumber, stepCount, dogState, dogIndex, episodeDone, pathSequence, lastUpdate, history, stepsPerEpisode, qSnapshots, params, stateMaps]);

    function configureEnvironment() {
        const nextConfig = {
            xMin,
            xMax,
            yMin,
            yMax,
            goalX: clamp(goalX, xMin, xMax),
            goalY: clamp(goalY, yMin, yMax),
            fixedStartX: clamp(fixedStartX, xMin, xMax),
            fixedStartY: clamp(fixedStartY, yMin, yMax),
            alpha,
            gamma,
            epsilon,
            rewardValue,
            randomStartEachEpisode,
        };
        const nextMaps = buildStateMaps(nextConfig);
        setAppliedConfig(nextConfig);
        applySnapshot(buildFreshSnapshot(nextConfig, nextMaps), { resetPlayback: true });
        setAutoplay(false);
    }

    function resetEpisode() {
        const startState = getEpisodeStartState(params);
        const startIndex = stateMaps.indexByKey.get(makeStateKey(startState.x, startState.y));
        applySnapshot(
            buildCurrentSnapshot({
                dogState: startState,
                dogIndex: startIndex,
                stepCount: 0,
                episodeDone: startState.x === params.goalX && startState.y === params.goalY,
                pathSequence: [startState],
                lastUpdate: null,
            }),
            { resetPlayback: true }
        );
    }

    function resetAll() {
        applySnapshot(buildFreshSnapshot(params, stateMaps), { resetPlayback: true });
        setAutoplay(false);
    }

    function stepEpisode() {
        const nextSnapshot = simulateOneStep(buildCurrentSnapshot(), params, stateMaps);
        applySnapshot(nextSnapshot, { resetPlayback: true });
    }

    function trainBatch(episodesToRun) {
        const nextSnapshot = simulateBatch(buildCurrentSnapshot(), params, stateMaps, episodesToRun);
        applySnapshot(nextSnapshot, { resetPlayback: true });
    }

    return (
        <div className="min-h-screen bg-slate-50 p-4 md:p-6">
            <div className="mx-auto max-w-7xl space-y-6">
                <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight">Q-Learning: Teach the Dog to Find the Bone on a 2D Grid</h1>
                    <p className="max-w-5xl text-slate-600">
                        This version extends the 1D idea to a rectangular world with four actions per cell. Watch sparse reward propagate backward from the goal, compare exploration against greedy choice, and inspect how ties in the learned policy naturally create diagonal summaries.
                    </p>
                </div>

                <div className="grid gap-6 md:grid-cols-[380px_1fr]">
                    <div className="space-y-6">
                        <Card className="rounded-2xl">
                            <CardHeader>
                                <CardTitle>Algorithm Parameters</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-5">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="space-y-2">
                                        <Label>Min X</Label>
                                        <Input type="number" value={xMin} onChange={(e) => {
                                            const v = Number(e.target.value);
                                            if (!Number.isNaN(v) && v < xMax) setXMin(v);
                                        }} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Max X</Label>
                                        <Input type="number" value={xMax} onChange={(e) => {
                                            const v = Number(e.target.value);
                                            if (!Number.isNaN(v) && v > xMin) setXMax(v);
                                        }} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Min Y</Label>
                                        <Input type="number" value={yMin} onChange={(e) => {
                                            const v = Number(e.target.value);
                                            if (!Number.isNaN(v) && v < yMax) setYMin(v);
                                        }} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Max Y</Label>
                                        <Input type="number" value={yMax} onChange={(e) => {
                                            const v = Number(e.target.value);
                                            if (!Number.isNaN(v) && v > yMin) setYMax(v);
                                        }} />
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-3">
                                    <div className="space-y-2">
                                        <Label>Goal X</Label>
                                        <Input type="number" value={goalX} onChange={(e) => {
                                            const v = Number(e.target.value);
                                            if (!Number.isNaN(v)) setGoalX(clamp(v, xMin, xMax));
                                        }} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Goal Y</Label>
                                        <Input type="number" value={goalY} onChange={(e) => {
                                            const v = Number(e.target.value);
                                            if (!Number.isNaN(v)) setGoalY(clamp(v, yMin, yMax));
                                        }} />
                                    </div>
                                </div>

                                {!randomStartEachEpisode && (
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="space-y-2">
                                            <Label>Fixed Start X</Label>
                                            <Input type="number" value={fixedStartX} onChange={(e) => {
                                                const v = Number(e.target.value);
                                                if (!Number.isNaN(v)) setFixedStartX(clamp(v, xMin, xMax));
                                            }} />
                                        </div>
                                        <div className="space-y-2">
                                            <Label>Fixed Start Y</Label>
                                            <Input type="number" value={fixedStartY} onChange={(e) => {
                                                const v = Number(e.target.value);
                                                if (!Number.isNaN(v)) setFixedStartY(clamp(v, yMin, yMax));
                                            }} />
                                        </div>
                                    </div>
                                )}

                                <ParamSlider label="Learning Rate (Alpha)" min={0} max={1} step={0.01} value={alpha} onChange={setAlpha} formatValue={(v) => v.toFixed(2)} />
                                <ParamSlider label="Discount Factor (Gamma)" min={0} max={0.99} step={0.01} value={gamma} onChange={setGamma} formatValue={(v) => v.toFixed(2)} />
                                <ParamSlider label="Exploration Rate (Epsilon)" min={0} max={1} step={0.01} value={epsilon} onChange={setEpsilon} formatValue={(v) => v.toFixed(2)} />
                                <ParamSlider label="Goal Reward" min={1} max={50} step={1} value={rewardValue} onChange={setRewardValue} />

                                <div className="flex items-center justify-between rounded-xl border p-3">
                                    <div>
                                        <Label>Random start each episode</Label>
                                        <p className="text-xs text-slate-500">Turn on for broader state coverage over the whole grid.</p>
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

                                {hasPendingConfig && (
                                    <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                                        Parameters changed. Click <span className="font-semibold">Configure</span> to rebuild the grid and restart training.
                                    </div>
                                )}

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

                                <div className="rounded-xl border p-3">
                                    <div className="mb-3 flex items-center justify-between">
                                        <div>
                                            <div className="font-medium">Playback Controls</div>
                                            <div className="text-xs text-slate-500">Inspect earlier states and update logs from the current training history.</div>
                                        </div>
                                        <Badge variant="outline">{playbackIndex == null ? "Live" : `Playback #${playbackIndex + 1}`}</Badge>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <Button variant="outline" size="sm" onClick={() => setPlaybackIndex((p) => (history.length === 0 ? null : p == null ? history.length - 1 : Math.max(0, p - 1)))} disabled={history.length === 0}>
                                            <ChevronLeft className="h-4 w-4" />
                                        </Button>
                                        <Button variant="outline" size="sm" onClick={() => setPlaybackIndex((p) => (history.length === 0 ? null : p == null ? 0 : Math.min(history.length - 1, p + 1)))} disabled={history.length === 0}>
                                            <ChevronRight className="h-4 w-4" />
                                        </Button>
                                        <Button variant="secondary" size="sm" onClick={() => setPlaybackIndex(null)} disabled={playbackIndex == null}>Return to Live</Button>
                                        <div className="text-xs text-slate-500">{history.length === 0 ? "No history yet." : `${history.length} saved states`}</div>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-3 text-sm">
                                    <div className="rounded-xl border p-3">
                                        <div className="text-slate-500">Completed Episodes</div>
                                        <div className="text-2xl font-bold">{completedEpisodes}</div>
                                    </div>
                                    <div className="rounded-xl border p-3">
                                        <div className="text-slate-500">Current Episode</div>
                                        <div className="text-2xl font-bold">{currentEpisodeNumber}</div>
                                    </div>
                                    <div className="rounded-xl border p-3">
                                        <div className="text-slate-500">Steps This Episode</div>
                                        <div className="text-2xl font-bold">{stepCount}</div>
                                    </div>
                                    <div className="rounded-xl border p-3">
                                        <div className="text-slate-500">Average Steps</div>
                                        <div className="text-2xl font-bold">{stepsPerEpisode.length ? (stepsPerEpisode.reduce((a, b) => a + b, 0) / stepsPerEpisode.length).toFixed(1) : "0.0"}</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    <div className="space-y-6">
                        <Card className="rounded-2xl">
                            <CardHeader>
                                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                    <CardTitle>2D Grid World</CardTitle>
                                    <div className="flex flex-wrap items-center gap-2 text-sm">
                                        <Badge variant="secondary">Live episode: {currentEpisodeNumber}</Badge>
                                        <Badge>{playbackFrame.note}</Badge>
                                        <Badge variant="outline">Goal: ({appliedConfig.goalX},{appliedConfig.goalY})</Badge>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid gap-4 md:grid-cols-[1fr_340px]">
                                    <div className="space-y-4">
                                        <div className="rounded-xl border p-4">
                                            <div className="mb-2 flex items-center justify-between">
                                                <h3 className="font-semibold">Current Path</h3>
                                                <span className="text-xs text-slate-500">Full route through this episode</span>
                                            </div>
                                            <div className="max-h-24 overflow-auto rounded-lg bg-slate-100 px-3 py-2 font-mono text-sm text-slate-700">
                                                {displayedPath}
                                            </div>
                                        </div>

                                        <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${appliedConfig.xMax - appliedConfig.xMin + 1}, minmax(0, 1fr))` }}>
                                            {stateMaps.states.map((state) => (
                                                <GridCell
                                                    key={state.key}
                                                    state={state}
                                                    dogState={playbackFrame.dogState}
                                                    goalState={{ x: appliedConfig.goalX, y: appliedConfig.goalY }}
                                                    isPlayback={playbackIndex != null}
                                                />
                                            ))}
                                        </div>
                                    </div>

                                    <div className="rounded-xl border p-4">
                                        <div className="mb-2 flex items-center justify-between">
                                            <h3 className="font-semibold">Optimal Action Field</h3>
                                            <span className="text-xs text-slate-500">Diagonal icons mean tied shortest directions</span>
                                        </div>
                                        <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${appliedConfig.xMax - appliedConfig.xMin + 1}, minmax(0, 1fr))` }}>
                                            {policyRows.map((row) => {
                                                const isGoal = row.x === appliedConfig.goalX && row.y === appliedConfig.goalY;
                                                return (
                                                    <div
                                                        key={`policy-${row.key}`}
                                                        className={[
                                                            "flex aspect-square min-w-0 flex-col items-center justify-center rounded-xl border shadow-sm",
                                                            isGoal ? "border-amber-400 bg-amber-100" : "border-sky-200 bg-sky-100",
                                                        ].join(" ")}
                                                    >
                                                        <PolicyIcon symbol={row.symbol} />
                                                        <div className="mt-1 text-[11px] text-slate-700">({row.x},{row.y})</div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        <div className="grid gap-6 xl:grid-cols-2">
                            <Card className="rounded-2xl">
                                <CardHeader>
                                    <CardTitle>Bellman Update Log</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    {!playbackFrame.update && <p className="text-sm text-slate-500">No update yet. Step once to see the exact algebra for a single Q-learning transition.</p>}
                                    {playbackFrame.update && (
                                        <>
                                            <div className="grid grid-cols-2 gap-3 text-sm">
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Episode</span><div className="text-lg font-semibold">{playbackFrame.update.episode}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Step in Episode</span><div className="text-lg font-semibold">{playbackFrame.update.stepInEpisode}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">State</span><div className="text-lg font-semibold">({playbackFrame.update.state.x},{playbackFrame.update.state.y})</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Next State</span><div className="text-lg font-semibold">({playbackFrame.update.nextState.x},{playbackFrame.update.nextState.y})</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Chosen Action</span><div className="text-lg font-semibold">{playbackFrame.update.actionShort} · {playbackFrame.update.actionLabel}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Mode</span><div className="text-lg font-semibold">{playbackFrame.update.mode}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Reward</span><div className="text-lg font-semibold">{playbackFrame.update.reward}</div></div>
                                                <div className="rounded-xl border p-3"><span className="text-slate-500">Best Next Action(s)</span><div className="text-lg font-semibold">{playbackFrame.update.bestNextActions.length ? playbackFrame.update.bestNextActions.join(", ") : "Terminal"}</div></div>
                                            </div>

                                            <div className="rounded-xl bg-slate-900 p-4 font-mono text-sm text-slate-100">
                                                <div>{playbackFrame.update.equation}</div>
                                                <div className="mt-3 text-sky-300">{playbackFrame.update.numericEquation}</div>
                                            </div>

                                            <div className="space-y-2 text-sm text-slate-700">
                                                <p><span className="font-semibold">Old Q:</span> {formatNum(playbackFrame.update.oldQ)}</p>
                                                <p><span className="font-semibold">Max next-state Q:</span> {formatNum(playbackFrame.update.maxNextQ)}</p>
                                                <p><span className="font-semibold">Target:</span> {formatNum(playbackFrame.update.target)}</p>
                                                <p><span className="font-semibold">New Q:</span> {formatNum(playbackFrame.update.newQ)}</p>
                                            </div>
                                        </>
                                    )}
                                </CardContent>
                            </Card>

                        </div>

                        <Card className="rounded-2xl">
                            <CardHeader>
                                <CardTitle>Q-Matrix</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-auto rounded-xl border">
                                    <table className="min-w-full text-sm">
                                        <thead className="bg-slate-100">
                                            <tr>
                                                <th className="px-4 py-3 text-left">State</th>
                                                <th className="px-4 py-3 text-left">U</th>
                                                <th className="px-4 py-3 text-left">D</th>
                                                <th className="px-4 py-3 text-left">L</th>
                                                <th className="px-4 py-3 text-left">R</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {(playbackFrame.qTable ?? qTable).map((row, idx) => {
                                                const state = stateMaps.states[idx];
                                                const isGoal = state.x === appliedConfig.goalX && state.y === appliedConfig.goalY;
                                                const symbol = getPolicySymbol(row.values, isGoal);
                                                const bestActions = getBestActionIndices(row.values);
                                                const isUnlearned = row.values.every((v) => v === 0);
                                                return (
                                                    <tr key={row.key} className={idx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                                                        <td className="px-4 py-3 font-medium">({state.x},{state.y}){isGoal ? " 🥩" : ""}</td>
                                                        {row.values.map((value, actionIdx) => (
                                                            <td key={actionIdx} className={["px-4 py-3 font-mono", !isUnlearned && bestActions.includes(actionIdx) ? "bg-emerald-50 text-emerald-700" : ""].join(" ")}>
                                                                {formatNum(value)}
                                                            </td>
                                                        ))}
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
    );
}
