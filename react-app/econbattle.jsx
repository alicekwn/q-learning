import React, { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

// ---------- pure math helpers ----------

function priceKey(p) {
    return Number(p).toFixed(3);
}

function stateIndex(p1, p2, prices) {
    const n = prices.length;
    const i = prices.findIndex((p) => Math.abs(p - p1) < 1e-6);
    const j = prices.findIndex((p) => Math.abs(p - p2) < 1e-6);
    if (i < 0 || j < 0) return -1;
    return i * n + j;
}

function indexToState(s, prices) {
    const n = prices.length;
    return [prices[Math.floor(s / n)], prices[s % n]];
}

function argmaxRow(row) {
    let best = 0;
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return best;
}

function demand1(p1, p2, k1, k2) { return Math.max(0, k1 - p1 + k2 * p2); }
function demand2(p1, p2, k1, k2) { return Math.max(0, k1 - p2 + k2 * p1); }
function profit1(p1, p2, c, k1, k2) { return (p1 - c) * demand1(p1, p2, k1, k2); }
function profit2(p1, p2, c, k1, k2) { return (p2 - c) * demand2(p1, p2, k1, k2); }

function fmt(x, d = 2) {
    return Number.isFinite(x) ? x.toFixed(d) : "—";
}

// Flip Q-table: s(p1,p2) -> s(p2,p1), so a Q2-trained table can act as Q1
function flipQTable(Q, prices) {
    const n = Q.length;
    const flipped = Q.map((row) => [...row]);
    for (let s = 0; s < n; s++) {
        const [p1, p2] = indexToState(s, prices);
        const sFlipped = stateIndex(p2, p1, prices);
        if (sFlipped >= 0) flipped[sFlipped] = [...Q[s]];
    }
    return flipped;
}

// Follow greedy policy until a state repeats (cycle detection)
function followGreedyUntilLoop(Q1, Q2, startP1, startP2, prices, maxSteps = 50000) {
    let s = stateIndex(startP1, startP2, prices);
    if (s < 0) return { path: [], loopStart: null, loop: [] };
    const firstSeen = new Map();
    const path = [];

    for (let t = 0; t < maxSteps; t++) {
        if (firstSeen.has(s)) {
            const loopStart = firstSeen.get(s);
            return { path, loopStart, loop: path.slice(loopStart) };
        }
        firstSeen.set(s, t);
        const a1 = argmaxRow(Q1[s]);
        const a2 = argmaxRow(Q2[s]);
        const p1Next = prices[a1];
        const p2Next = prices[a2];
        const sNext = stateIndex(p1Next, p2Next, prices);
        path.push({ t, a1Price: p1Next, a2Price: p2Next });
        s = sNext;
    }
    return { path, loopStart: null, loop: [] };
}

// Parse CSV text: first column is state label (quoted or not), rest are Q-values.
// Returns { Q: number[][], prices: number[] } or throws.
function parseQTableCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) throw new Error("CSV has too few lines.");

    // Parse a CSV line respecting quoted fields
    function parseLine(line) {
        const fields = [];
        let cur = "";
        let inQuote = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') { inQuote = !inQuote; continue; }
            if (ch === "," && !inQuote) { fields.push(cur); cur = ""; continue; }
            cur += ch;
        }
        fields.push(cur);
        return fields;
    }

    const headerFields = parseLine(lines[0]);
    // Price columns: format "p=X.X" (from econ.jsx export)
    const prices = [];
    for (const f of headerFields.slice(1)) {
        const m = f.trim().match(/^p=(.+)$/);
        if (m) prices.push(parseFloat(m[1]));
    }
    if (prices.length === 0) throw new Error('No price columns found. Expected header format: "p=X.X".');

    const Q = [];
    for (let i = 1; i < lines.length; i++) {
        const fields = parseLine(lines[i]);
        if (fields.length < 2) continue;
        const row = fields.slice(1).map((v) => parseFloat(v.trim()));
        if (row.some(Number.isNaN)) throw new Error(`Non-numeric value on line ${i + 1}.`);
        Q.push(row);
    }

    const nStates = prices.length ** 2;
    if (Q.length !== nStates) {
        throw new Error(`Expected ${nStates} state rows (m²), got ${Q.length}.`);
    }
    if (Q.some((row) => row.length !== prices.length)) {
        throw new Error("Row length doesn't match number of price columns.");
    }

    return { Q, prices };
}

// ---------- sub-components ----------

function QMatrixDisplay({ Q, prices, title }) {
    const nDisplay = Math.min(Q.length, 20); // show first 20 rows for brevity
    return (
        <div>
            <div className="mb-1 text-xs font-semibold text-slate-600">{title}</div>
            <div className="max-h-52 overflow-auto rounded-xl border">
                <table className="min-w-full text-xs">
                    <thead className="bg-slate-100 sticky top-0">
                        <tr>
                            <th className="px-2 py-1 text-left font-medium">State</th>
                            {prices.map((p) => (
                                <th key={p} className="px-2 py-1 text-left font-mono font-medium">
                                    p={p.toFixed(1)}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {Q.slice(0, nDisplay).map((row, idx) => {
                            const [p1, p2] = indexToState(idx, prices);
                            const best = argmaxRow(row);
                            return (
                                <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                                    <td className="px-2 py-1 font-medium whitespace-nowrap">
                                        s({p1.toFixed(1)},{p2.toFixed(1)})
                                    </td>
                                    {row.map((v, j) => (
                                        <td
                                            key={j}
                                            className={["px-2 py-1 font-mono", best === j ? "bg-emerald-50 text-emerald-700 font-semibold" : ""].join(" ")}
                                        >
                                            {v.toFixed(2)}
                                        </td>
                                    ))}
                                </tr>
                            );
                        })}
                        {Q.length > nDisplay && (
                            <tr>
                                <td colSpan={prices.length + 1} className="px-2 py-1 text-center text-slate-400 text-xs">
                                    … {Q.length - nDisplay} more rows
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function FileUploadPanel({ label, playerName, onPlayerNameChange, onLoad, loaded, error }) {
    const handleFile = useCallback((e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => onLoad(ev.target.result, file.name);
        reader.readAsText(file);
    }, [onLoad]);

    return (
        <div className="space-y-3">
            <div className="space-y-1">
                <Label className="text-xs">Player name</Label>
                <Input
                    className="text-sm"
                    value={playerName}
                    onChange={(e) => onPlayerNameChange(e.target.value)}
                    placeholder="e.g. Alice"
                />
            </div>
            <div className="space-y-1">
                <Label className="text-xs">{label}</Label>
                <input
                    type="file"
                    accept=".csv"
                    onChange={handleFile}
                    className="block w-full text-xs text-slate-600 file:mr-2 file:rounded file:border-0 file:bg-slate-100 file:px-2 file:py-1 file:text-xs file:font-medium hover:file:bg-slate-200"
                />
            </div>
            {error && <p className="text-xs text-red-600">{error}</p>}
            {loaded && !error && (
                <Badge variant="secondary" className="text-xs">
                    Loaded: {loaded.filename} ({loaded.prices.length} prices, {loaded.Q.length} states)
                </Badge>
            )}
        </div>
    );
}

// ---------- main component ----------

export default function EconBattle() {
    const [player1Name, setPlayer1Name] = useState("Alice");
    const [player2Name, setPlayer2Name] = useState("Bob");
    const [loaded1, setLoaded1] = useState(null); // { Q, prices, filename }
    const [loaded2, setLoaded2] = useState(null);
    const [error1, setError1] = useState(null);
    const [error2, setError2] = useState(null);
    const [persp1, setPersp1] = useState("Q1"); // "Q1" | "Q2"
    const [persp2, setPersp2] = useState("Q2");

    // Battle env params
    const [k1, setK1] = useState(7);
    const [k2, setK2] = useState(0.5);
    const [c, setC] = useState(2);
    const [startP1, setStartP1] = useState(7);
    const [startP2, setStartP2] = useState(7);

    const [result, setResult] = useState(null); // computed battle result
    const [battleError, setBattleError] = useState(null);
    const [customPaths, setCustomPaths] = useState([{ id: 0, aliceInput: "", bobInput: "" }]);
    const [nextCustomId, setNextCustomId] = useState(1);

    function handleLoad1(text, filename) {
        setError1(null);
        setLoaded1(null);
        setResult(null);
        try {
            const { Q, prices } = parseQTableCSV(text);
            setLoaded1({ Q, prices, filename });
        } catch (e) {
            setError1(e.message);
        }
    }

    function handleLoad2(text, filename) {
        setError2(null);
        setLoaded2(null);
        setResult(null);
        try {
            const { Q, prices } = parseQTableCSV(text);
            setLoaded2({ Q, prices, filename });
        } catch (e) {
            setError2(e.message);
        }
    }

    function computeBattle() {
        setBattleError(null);
        setResult(null);

        if (!loaded1 || !loaded2) {
            setBattleError("Please upload both Q-table CSV files.");
            return;
        }

        // Use prices from file 1; warn if they differ
        const prices = loaded1.prices;
        if (JSON.stringify(prices) !== JSON.stringify(loaded2.prices)) {
            setBattleError(
                `Price sets differ. File 1: [${loaded1.prices.map((p) => p.toFixed(1)).join(", ")}]` +
                ` | File 2: [${loaded2.prices.map((p) => p.toFixed(1)).join(", ")}]`
            );
            return;
        }

        // Normalize start prices
        const normP1 = prices.reduce((best, p) => Math.abs(p - startP1) < Math.abs(best - startP1) ? p : best, prices[0]);
        const normP2 = prices.reduce((best, p) => Math.abs(p - startP2) < Math.abs(best - startP2) ? p : best, prices[0]);

        if (!prices.some((p) => Math.abs(p - normP1) < 1e-6) || !prices.some((p) => Math.abs(p - normP2) < 1e-6)) {
            setBattleError(`Starting prices must be close to a valid price. Valid: [${prices.map((p) => p.toFixed(1)).join(", ")}]`);
            return;
        }

        // Apply perspective flipping
        let Q1 = loaded1.Q;
        let Q2 = loaded2.Q;
        if (persp1 === "Q2") Q1 = flipQTable(Q1, prices); // uploaded as Q2, used as Q1 → flip
        if (persp2 === "Q1") Q2 = flipQTable(Q2, prices); // uploaded as Q1, used as Q2 → flip

        // Compute equilibrium & collusion reference values
        const pe = (k1 + c) / (2 - k2);
        const pc = (2 * k1 + 2 * c * (1 - k2)) / (4 * (1 - k2));
        const piE = profit1(pe, pe, c, k1, k2);
        const piC = profit1(pc, pc, c, k1, k2);

        const traj = followGreedyUntilLoop(Q1, Q2, normP1, normP2, prices, 50000);

        // Compute per-step profits for full path (with step 0 = starting prices)
        const fullPath = [
            { a1Price: normP1, a2Price: normP2 },
            ...traj.path.map((r) => ({ a1Price: r.a1Price, a2Price: r.a2Price })),
        ];
        const loopStartFull = traj.loopStart != null ? traj.loopStart + 1 : null;

        let avgPi1 = null, avgPi2 = null, avgP1 = null, avgP2 = null;
        let norm1 = null, norm2 = null;

        if (traj.loop.length > 0) {
            let sumPi1 = 0, sumPi2 = 0, sumP1 = 0, sumP2 = 0;
            for (const rec of traj.loop) {
                sumPi1 += profit1(rec.a1Price, rec.a2Price, c, k1, k2);
                sumPi2 += profit2(rec.a1Price, rec.a2Price, c, k1, k2);
                sumP1 += rec.a1Price;
                sumP2 += rec.a2Price;
            }
            avgPi1 = sumPi1 / traj.loop.length;
            avgPi2 = sumPi2 / traj.loop.length;
            avgP1 = sumP1 / traj.loop.length;
            avgP2 = sumP2 / traj.loop.length;
            const denom = piC - piE;
            if (Math.abs(denom) > 1e-10) {
                norm1 = (avgPi1 - piE) / denom;
                norm2 = (avgPi2 - piE) / denom;
            }
        }

        setResult({
            fullPath,
            loopStartFull,
            loopLength: traj.loop.length,
            avgPi1, avgPi2, avgP1, avgP2, norm1, norm2,
            pe, pc, piE, piC,
            prices,
            Q1, Q2,
        });
    }

    // Determine winner text
    let winnerBadge = null;
    if (result && result.avgPi1 != null) {
        if (result.avgPi1 > result.avgPi2) winnerBadge = { text: `${player1Name} wins!`, color: "bg-emerald-100 text-emerald-800 border-emerald-200" };
        else if (result.avgPi2 > result.avgPi1) winnerBadge = { text: `${player2Name} wins!`, color: "bg-emerald-100 text-emerald-800 border-emerald-200" };
        else winnerBadge = { text: "Tie!", color: "bg-slate-100 text-slate-800 border-slate-200" };
    }

    return (
        <div className="min-h-screen bg-slate-50 p-4 md:p-6">
            <div className="mx-auto space-y-6">
                <div className="space-y-1">
                    <h1 className="text-3xl font-bold tracking-tight">Pricing Battle</h1>
                    <p className="text-slate-600 max-w-3xl">
                        Upload Q-tables exported from the training page (CSV). The battle follows the greedy policy of each Q-table and detects the pricing cycle, then declares a winner by average profit.
                    </p>
                </div>

                {/* Upload panels */}
                <div className="grid gap-6 md:grid-cols-2">
                    <Card className="rounded-2xl">
                        <CardHeader><CardTitle>Player 1 Q-Table</CardTitle></CardHeader>
                        <CardContent className="space-y-4">
                            <FileUploadPanel
                                label="Upload Q-table CSV"
                                playerName={player1Name}
                                onPlayerNameChange={setPlayer1Name}
                                onLoad={handleLoad1}
                                loaded={loaded1}
                                error={error1}
                            />
                            {loaded1 && (
                                <>
                                    <div className="space-y-1">
                                        <Label className="text-xs">This Q-table was trained from the perspective of:</Label>
                                        <div className="flex gap-2">
                                            {["Q1", "Q2"].map((opt) => (
                                                <button
                                                    key={opt}
                                                    type="button"
                                                    onClick={() => setPersp1(opt)}
                                                    className={[
                                                        "rounded-lg border px-3 py-1.5 text-xs font-medium transition",
                                                        persp1 === opt ? "border-slate-800 bg-slate-800 text-white" : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
                                                    ].join(" ")}
                                                >
                                                    Player {opt === "Q1" ? "1" : "2"} ({opt})
                                                </button>
                                            ))}
                                        </div>
                                        {persp1 === "Q2" && (
                                            <p className="text-xs text-amber-600">States will be flipped (p1↔p2) before use as Player 1.</p>
                                        )}
                                    </div>
                                    <QMatrixDisplay Q={loaded1.Q} prices={loaded1.prices} title={`Q-table preview (${loaded1.filename})`} />
                                </>
                            )}
                        </CardContent>
                    </Card>

                    <Card className="rounded-2xl">
                        <CardHeader><CardTitle>Player 2 Q-Table</CardTitle></CardHeader>
                        <CardContent className="space-y-4">
                            <FileUploadPanel
                                label="Upload Q-table CSV"
                                playerName={player2Name}
                                onPlayerNameChange={setPlayer2Name}
                                onLoad={handleLoad2}
                                loaded={loaded2}
                                error={error2}
                            />
                            {loaded2 && (
                                <>
                                    <div className="space-y-1">
                                        <Label className="text-xs">This Q-table was trained from the perspective of:</Label>
                                        <div className="flex gap-2">
                                            {["Q1", "Q2"].map((opt) => (
                                                <button
                                                    key={opt}
                                                    type="button"
                                                    onClick={() => setPersp2(opt)}
                                                    className={[
                                                        "rounded-lg border px-3 py-1.5 text-xs font-medium transition",
                                                        persp2 === opt ? "border-slate-800 bg-slate-800 text-white" : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
                                                    ].join(" ")}
                                                >
                                                    Player {opt === "Q1" ? "1" : "2"} ({opt})
                                                </button>
                                            ))}
                                        </div>
                                        {persp2 === "Q1" && (
                                            <p className="text-xs text-amber-600">States will be flipped (p1↔p2) before use as Player 2.</p>
                                        )}
                                    </div>
                                    <QMatrixDisplay Q={loaded2.Q} prices={loaded2.prices} title={`Q-table preview (${loaded2.filename})`} />
                                </>
                            )}
                        </CardContent>
                    </Card>
                </div>

                {/* Environment params */}
                <Card className="rounded-2xl">
                    <CardHeader>
                        <CardTitle>Environment Parameters</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="mb-3 text-xs text-amber-700 border border-amber-200 bg-amber-50 rounded-lg px-3 py-2">
                            ⚠️ These must match the parameters used to train the uploaded Q-tables.
                        </p>
                        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
                            {[
                                { label: "k1", value: k1, set: setK1, step: 0.1 },
                                { label: "k2", value: k2, set: setK2, step: 0.01 },
                                { label: "c (Marginal Cost)", value: c, set: setC, step: 0.1 },
                                { label: "Starting p1", value: startP1, set: setStartP1, step: 0.1 },
                                { label: "Starting p2", value: startP2, set: setStartP2, step: 0.1 },
                            ].map(({ label, value, set, step }) => (
                                <div key={label} className="space-y-1">
                                    <Label className="text-xs">{label}</Label>
                                    <Input type="number" step={step} value={value} onChange={(e) => set(Number(e.target.value))} />
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                {/* Compute button */}
                <Button className="w-full" size="lg" onClick={computeBattle} disabled={!loaded1 || !loaded2}>
                    Compute Trajectory & Determine Winner
                </Button>

                {battleError && (
                    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                        ❌ {battleError}
                    </div>
                )}

                {/* Results */}
                {result && (
                    <div className="space-y-6">
                        {/* Reference values */}
                        <Card className="rounded-2xl">
                            <CardContent className="pt-4">
                                <div className="flex flex-wrap gap-4 text-sm">
                                    <span>Equilibrium price: <strong>{fmt(result.pe)}</strong></span>
                                    <span>Collusion price: <strong>{fmt(result.pc)}</strong></span>
                                    <span>π_e: <strong>{fmt(result.piE)}</strong></span>
                                    <span>π_c: <strong>{fmt(result.piC)}</strong></span>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Winner banner */}
                        {winnerBadge && result.loopStartFull != null && (
                            <div className={`rounded-2xl border px-6 py-4 text-lg font-bold ${winnerBadge.color}`}>
                                🏆 {winnerBadge.text}
                            </div>
                        )}

                        {/* Profit summary */}
                        {result.avgPi1 != null && (
                            <div className="grid gap-4 md:grid-cols-2">
                                {[
                                    { name: player1Name, avgP: result.avgP1, avgPi: result.avgPi1, norm: result.norm1 },
                                    { name: player2Name, avgP: result.avgP2, avgPi: result.avgPi2, norm: result.norm2 },
                                ].map((p, i) => (
                                    <Card key={i} className="rounded-2xl">
                                        <CardContent className="pt-4 space-y-1 text-sm">
                                            <div className="font-semibold text-base">{p.name} (Player {i + 1})</div>
                                            <div>Average price p̄{i + 1} = <strong>{fmt(p.avgP, 3)}</strong></div>
                                            <div>Average profit π̄{i + 1} = <strong>{fmt(p.avgPi, 4)}</strong></div>
                                            <div>Normalised Δ{i + 1} = <strong>{p.norm != null ? fmt(p.norm, 3) : "N/A"}</strong></div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        )}

                        {/* Trajectory table — prices */}
                        <Card className="rounded-2xl">
                            <CardHeader>
                                <CardTitle>
                                    Price Trajectory
                                    {result.loopStartFull != null
                                        ? ` — cycle starts at step ${result.loopStartFull - 1}, length ${result.loopLength}`
                                        : " — no cycle detected"}
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto rounded-xl border">
                                    <table className="border-collapse text-xs" style={{ tableLayout: "fixed" }}>
                                        <tbody>
                                            <tr className="bg-white">
                                                <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-b border-slate-200 w-20 min-w-20">{player1Name}</td>
                                                {result.fullPath.map((rec, i) => (
                                                    <td key={i} className={["px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14",
                                                        result.loopStartFull != null && i >= result.loopStartFull ? "bg-blue-50 text-blue-800" : ""].join(" ")}>
                                                        {rec.a1Price.toFixed(1)}
                                                    </td>
                                                ))}
                                            </tr>
                                            <tr className="bg-slate-50">
                                                <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-b border-slate-200 w-20 min-w-20">{player2Name}</td>
                                                {result.fullPath.map((rec, i) => (
                                                    <td key={i} className={["px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14",
                                                        result.loopStartFull != null && i >= result.loopStartFull ? "bg-blue-50 text-blue-800" : ""].join(" ")}>
                                                        {rec.a2Price.toFixed(1)}
                                                    </td>
                                                ))}
                                            </tr>
                                            <tr className="bg-white">
                                                <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-slate-200 w-20 min-w-20 text-slate-400">Step</td>
                                                {result.fullPath.map((_, i) => (
                                                    <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-400">{i}</td>
                                                ))}
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                {result.loopStartFull != null && <p className="mt-1 text-xs text-blue-600">Blue = cycle</p>}
                                {result.fullPath.length > 10 && <p className="mt-1 text-xs text-slate-400">Scroll horizontally to see all steps.</p>}
                            </CardContent>
                        </Card>

                        {/* Trajectory table — profits */}
                        <Card className="rounded-2xl">
                            <CardHeader><CardTitle>Profit Trajectory</CardTitle></CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto rounded-xl border">
                                    <table className="border-collapse text-xs" style={{ tableLayout: "fixed" }}>
                                        <tbody>
                                            <tr className="bg-white">
                                                <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-b border-slate-200 w-20 min-w-20">π1 ({player1Name})</td>
                                                {result.fullPath.map((rec, i) => (
                                                    <td key={i} className={["px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14",
                                                        result.loopStartFull != null && i >= result.loopStartFull ? "bg-blue-50 text-blue-800" : ""].join(" ")}>
                                                        {fmt(profit1(rec.a1Price, rec.a2Price, c, k1, k2))}
                                                    </td>
                                                ))}
                                            </tr>
                                            <tr className="bg-slate-50">
                                                <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-b border-slate-200 w-20 min-w-20">π2 ({player2Name})</td>
                                                {result.fullPath.map((rec, i) => (
                                                    <td key={i} className={["px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14",
                                                        result.loopStartFull != null && i >= result.loopStartFull ? "bg-blue-50 text-blue-800" : ""].join(" ")}>
                                                        {fmt(profit2(rec.a1Price, rec.a2Price, c, k1, k2))}
                                                    </td>
                                                ))}
                                            </tr>
                                            <tr className="bg-white">
                                                <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-slate-200 w-20 min-w-20 text-slate-400">Step</td>
                                                {result.fullPath.map((_, i) => (
                                                    <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-400">{i}</td>
                                                ))}
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                {result.loopStartFull != null && <p className="mt-1 text-xs text-blue-600">Blue = cycle</p>}
                                {result.fullPath.length > 10 && <p className="mt-1 text-xs text-slate-400">Scroll horizontally to see all steps.</p>}
                            </CardContent>
                        </Card>

                        {/* Custom Price Path Comparison */}
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
                                    Enter comma-separated prices for each player (e.g. <span className="font-mono">7, 8, 7, 8</span>). Profits are averaged over the entered sequence.
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
                                            sumPi1 += profit1(alicePrices[i], bobPrices[i], c, k1, k2);
                                            sumPi2 += profit2(alicePrices[i], bobPrices[i], c, k1, k2);
                                            sumP1 += alicePrices[i];
                                            sumP2 += bobPrices[i];
                                        }
                                        avgPi1 = sumPi1 / len;
                                        avgPi2 = sumPi2 / len;
                                        avgP1 = sumP1 / len;
                                        avgP2 = sumP2 / len;
                                        const denom = result.piC - result.piE;
                                        if (Math.abs(denom) > 1e-10) {
                                            norm1 = (avgPi1 - result.piE) / denom;
                                            norm2 = (avgPi2 - result.piE) / denom;
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
                                                    <Label className="text-xs">{player1Name} prices (p1)</Label>
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
                                                    <Label className="text-xs">{player2Name} prices (p2)</Label>
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
                                                                    <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-b border-slate-200 w-20 min-w-20">{player1Name}</td>
                                                                    {alicePrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14">{p.toFixed(1)}</td>
                                                                    ))}
                                                                </tr>
                                                                <tr className="bg-slate-50">
                                                                    <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-b border-slate-200 w-20 min-w-20">{player2Name}</td>
                                                                    {bobPrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-b border-r border-slate-100 w-14 min-w-14">{p.toFixed(1)}</td>
                                                                    ))}
                                                                </tr>
                                                                <tr className="bg-white">
                                                                    <td className="sticky left-0 z-10 bg-white px-3 py-2 font-semibold border-r border-slate-200 w-20 min-w-20 text-slate-400">π1</td>
                                                                    {alicePrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-600">
                                                                            {fmt(profit1(p, bobPrices[i], c, k1, k2))}
                                                                        </td>
                                                                    ))}
                                                                </tr>
                                                                <tr className="bg-slate-50">
                                                                    <td className="sticky left-0 z-10 bg-slate-50 px-3 py-2 font-semibold border-r border-slate-200 w-20 min-w-20 text-slate-400">π2</td>
                                                                    {bobPrices.slice(0, len).map((p, i) => (
                                                                        <td key={i} className="px-3 py-2 text-center border-r border-slate-100 w-14 min-w-14 text-slate-600">
                                                                            {fmt(profit2(alicePrices[i], p, c, k1, k2))}
                                                                        </td>
                                                                    ))}
                                                                </tr>
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2 space-y-0.5">
                                                            <div className="font-semibold text-slate-700">{player1Name} avg</div>
                                                            <div>p̄1 = {fmt(avgP1, 3)}</div>
                                                            <div>π̄1 = {fmt(avgPi1, 4)}</div>
                                                            {norm1 != null && <div className="text-slate-500">Δ1 = {fmt(norm1, 3)}</div>}
                                                        </div>
                                                        <div className="rounded-lg bg-slate-50 px-3 py-2 space-y-0.5">
                                                            <div className="font-semibold text-slate-700">{player2Name} avg</div>
                                                            <div>p̄2 = {fmt(avgP2, 3)}</div>
                                                            <div>π̄2 = {fmt(avgPi2, 4)}</div>
                                                            {norm2 != null && <div className="text-slate-500">Δ2 = {fmt(norm2, 3)}</div>}
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

                        {/* No cycle warning */}
                        {result.loopStartFull == null && (
                            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                                ⚠️ No cycle detected within 50,000 steps. Try different starting prices.
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
