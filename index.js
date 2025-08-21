// index.js — Zephyr (plain text) + HF toxicity + tagging + time-decayed aggregation
import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());
app.use(express.static("public"));

// ===== Config =====
const HF_TOKEN     = process.env.HF_TOKEN || ""; // toxicity only
const TOXICITY_MODEL = "unitary/unbiased-toxic-roberta";

const ZEPHYR_URL   = process.env.ZEPHYR_URL   || "http://192.168.1.243:1234/v1/chat/completions"; // LM Studio
const ZEPHYR_MODEL = process.env.ZEPHYR_MODEL || "zephyr-7b-beta";
const Z_TEMP = 0.2;
const Z_MAX  = 180;

// ===== Helpers =====
const PROFS = /\b(fuck\w*|shit\w*|bitch\w*|asshole\w*|crap\w*|damn\w*|ugly as fuck)\b/gi;

function perspectiveOf(text) {
  const t = (text || "").trim();
  return /\bI\b|\bI\s+think\b|\bIMO\b/i.test(t) ? "first-person" : "third-person";
}
function stripNoise(s) {
  if (!s) return s;
  return s
    .replace(/#[\w-]+/g, "")                                 // hashtags
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "") // emojis & symbols
    .replace(/[!]{2,}/g, "!")                                 // collapse !!!
    .replace(/\s{2,}/g, " ")
    .trim();
}
function cap2(s) { return (s || "").split(/(?<=[.?!])\s+/).slice(0,2).join(" ").trim(); }

// tiny rule-based backup so you never get empty output
function rbFeedback(text, perspective) {
  const t = (text || "").toLowerCase();
  const friend = (a,b) => cap2(`${a}. ${b}`);
  const third  = (m) => `This person thinks ${m}.`;
  const first  = (m) => `I think ${m}.`;
  const say    = (m) => (perspective === "first-person" ? first(m) : third(m));

  if (/\bbeard|stubble|mustache\b/.test(t)) return friend(say("the beard looks a bit unkept"), "Try a closer trim or line the edges");
  if (/\b(hair|fringe|bangs)\b/.test(t))   return friend(say("the hair looks a bit messy"), "Try brushing it back or a cleaner style");
  if (/\b(glasses?|frames?)\b/.test(t))    return friend(say("the glasses don’t flatter your face"), "Consider thinner frames or trying contacts");
  if (/\b(light(ing)?|dark|shadow)\b/.test(t)) return friend(say("the lighting isn’t very flattering"), "Take a shot near a window or outside");
  if (/\b(outfit|clothes|shirt|hoodie|jacket|fit)\b/.test(t)) return friend(say("the outfit reads a bit casual"), "A fitted tee or collared shirt can look sharper");
  return friend(say("a small tweak could help"), "Try one specific change to sharpen the look");
}

// ===== Tagging (taxonomy) =====
const TAGS = [
  { key: "photo:lighting",    re: /\b(light(ing)?|dark|shadow|harsh)\b/i },
  { key: "photo:blurry",      re: /\b(blur|blurr?y?|grain|low\s*res|noisy)\b/i },
  { key: "style:glasses",     re: /\b(glasses?|frames?)\b/i },
  { key: "grooming:hair",     re: /\b(hair|fringe|bangs)\b/i },
  { key: "grooming:beard",    re: /\b(beard|stubble|mustache)\b/i },
  { key: "style:outfit",      re: /\b(outfit|clothes|shirt|hoodie|jacket|fit)\b/i },
  { key: "photo:background",  re: /\b(background|messy\s*room|clutter)\b/i },
  { key: "photo:angle",       re: /\b(selfie|mirror|angle|bathroom)\b/i },
  { key: "expression:smile",  re: /\b(smile|expression)\b/i },
  { key: "bio:wording",       re: /\b(bio|profile|about|prompt)\b/i }
];
function detectTags(raw, feedback) {
  const text = `${raw || ""} ${feedback || ""}`;
  const found = new Set();
  for (const t of TAGS) if (t.re.test(text)) found.add(t.key);
  return Array.from(found.size ? found : new Set(["general"]));
}

// ===== Aggregation with time-decay (in-memory) =====
// profileId -> tag -> { score, updatedAtDays }
const agg = Object.create(null);
const HALF_LIFE_DAYS = 14;
const DECAY_BASE = Math.log(2) / HALF_LIFE_DAYS;
const nowDays = () => Date.now() / 86400000;

function addMentions(profileId, tags) {
  const tnow = nowDays();
  agg[profileId] ||= Object.create(null);
  for (const tag of tags) {
    const slot = agg[profileId][tag] || { score: 0, updatedAt: tnow };
    const dt = Math.max(0, tnow - slot.updatedAt);
    slot.score = slot.score * Math.exp(-DECAY_BASE * dt);
    slot.score += 1;
    slot.updatedAt = tnow;
    agg[profileId][tag] = slot;
  }
}
function topMentions(profileId, limit = 8) {
  const tnow = nowDays();
  const row = agg[profileId] || {};
  const items = Object.entries(row).map(([tag, slot]) => {
    const dt = Math.max(0, tnow - slot.updatedAt);
    const decayed = slot.score * Math.exp(-DECAY_BASE * dt);
    return { tag, score: decayed };
  });
  items.sort((a,b) => b.score - a.score);
  return items.slice(0, limit).map(it => ({
    tag: it.tag,
    score: Number(it.score.toFixed(2)),
    count: Math.max(1, Math.round(it.score))
  }));
}

// ===== Toxicity (HF) =====
async function hfToxicity(text) {
  if (!HF_TOKEN) return 0;
  try {
    const res = await fetch(`https://api-inference.huggingface.co/models/${TOXICITY_MODEL}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: text })
    });
    if (!res.ok) return 0;
    const data = await res.json();
    const arr = Array.isArray(data) && Array.isArray(data[0]) ? data[0] : data;
    if (!Array.isArray(arr)) return 0;
    const toxicLabels = ["toxic","severe_toxic","obscene","insult","threat","identity_hate","harassment","hate"];
    let max = 0;
    for (const s of arr) {
      const lbl = (s.label || "").toLowerCase();
      if (toxicLabels.some(t => lbl.includes(t))) max = Math.max(max, s.score || 0);
    }
    return max || 0;
  } catch {
    return 0;
  }
}

// ===== Zephyr (plain text) with 2 retries =====
function buildPromptPlain(text, perspective) {
  return [
    "Rewrite the comment as short, kind, casual feedback for a dating app.",
    `Keep the SAME perspective: ${perspective}. If the author uses first-person, keep first-person; otherwise use third-person (e.g., "This person thinks …").`,
    "Rules: at most TWO sentences. No emojis, no hashtags, no greetings, no exclamation marks. Remove profanity; keep the meaning. Focus on changeable style/behavior (clothes, photo, grooming, wording)—not identity/body.",
    "",
    `Comment: ${JSON.stringify(text)}`,
    "Feedback:"
  ].join("\n");
}
async function zephyrPlain(prompt, retries = 2) {
  let last = "";
  for (let i = 0; i <= retries; i++) {
    const res = await fetch(ZEPHYR_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: ZEPHYR_MODEL,
        messages: [
          { role: "system", content: "You reply only with the final feedback text. No lists, no headings, no emojis/hashtags." },
          { role: "user", content: prompt }
        ],
        temperature: Z_TEMP,
        max_tokens: Z_MAX
      })
    });
    if (!res.ok) { last = `HTTP ${res.status}`; continue; }
    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content || "";
    const cleaned  = cap2(stripNoise(content));
    if (cleaned) return cleaned;
    last = content;
  }
  throw new Error(last || "empty");
}

// ===== API =====
app.get("/health", (req, res) => {
  res.json({ ok: true, zephyr: { url: ZEPHYR_URL, model: ZEPHYR_MODEL }, toxicityModel: TOXICITY_MODEL });
});

app.post("/rewrite", async (req, res) => {
  const { text, profileId = "demo" } = req.body;
  if (!text || !text.trim()) return res.status(400).json({ error: "empty_text" });

  try {
    const tox = await hfToxicity(text);
    const toxicity = Math.round((tox || 0) * 100);
    const perspective = perspectiveOf(text);

    let feedback;
    try {
      const prompt = buildPromptPlain(text, perspective);
      feedback = await zephyrPlain(prompt, 2);
    } catch {
      const scrubbed = stripNoise(text.replace(PROFS, "").replace(/\s+/g, " ").trim());
      const rb = rbFeedback(text, perspective);
      feedback = cap2(rb || scrubbed || "Consider one small, specific tweak to improve the impression.");
    }

    const tags = detectTags(text, feedback);
    addMentions(profileId, tags);
    const mentions = topMentions(profileId);

    res.json({ toxicity, detoxText: feedback, feedback, tags, mentions, generator: `${ZEPHYR_MODEL} (plain-text)` });
  } catch (e) {
    console.error(e);
    res.status(502).json({ error: String(e?.message || e) });
  }
});

app.listen(3002, () => {
  console.log("✨ Server running on http://localhost:3002");
});
