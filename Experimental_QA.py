# app.py
import streamlit as st

st.set_page_config(page_title="Voice form (bank dataset)", page_icon="🗣️", layout="centered")
st.title("🗣️ Voice Form — Bank Dataset Fields")

st.markdown(
    """
Click **Start** once to allow mic + audio.  
I’ll ask you each field from the bank dataset and confirm what I heard.  
If I don’t catch it, I’ll ask again.
    """
)

btn = st.button("▶️ Start")

html = """
<div id="status" style="font-family:system-ui;margin:.75rem 0;font-size:1.05rem;">Ready.</div>
<div id="log" style="font-family:system-ui;padding:.75rem;background:#f6f6f6;border-radius:.5rem;max-height:280px;overflow:auto;white-space:pre-wrap;"></div>
<div id="summary" style="display:none;margin-top:12px;padding:.75rem;border:1px solid #ddd;border-radius:.5rem;"></div>

<script>
(async () => {
  const status = document.getElementById("status");
  const log = document.getElementById("log");
  const summary = document.getElementById("summary");

  function say(t) {
    return new Promise((resolve) => {
      const u = new SpeechSynthesisUtterance(t);
      u.onend = resolve;
      window.speechSynthesis.speak(u);
    });
  }

  function listenOnce(timeoutMs=15000) {
    return new Promise((resolve, reject) => {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) return reject(new Error("SpeechRecognition API not supported."));
      const r = new SR();
      r.lang = "en-US";
      r.interimResults = false;
      r.maxAlternatives = 1;

      const to = setTimeout(() => { try{r.abort();}catch(_){ }; reject(new Error("timeout")); }, timeoutMs);
      r.onresult = (e) => { clearTimeout(to); resolve((e.results[0][0].transcript || "").trim()); };
      r.onerror = (e) => { clearTimeout(to); reject(new Error(e.error || "recog error")); };
      try { r.start(); } catch (e) { clearTimeout(to); reject(e); }
    });
  }

  function addLog(line) {
    log.textContent += line + "\\n";
    log.scrollTop = log.scrollHeight;
  }

  // --- Helpers ---
  const yn = (t) => /^(y|yeah|yes|yup|true|1)$/i.test(t)? "yes"
                    : (/^(n|no|nope|false|0)$/i.test(t)? "no" : null);

  const monthsMap = {
    "january":"jan","february":"feb","march":"mar","april":"apr","may":"may","june":"jun",
    "july":"jul","august":"aug","september":"sep","october":"oct","november":"nov","december":"dec",
    "jan":"jan","feb":"feb","mar":"mar","apr":"apr","jun":"jun","jul":"jul","aug":"aug","sep":"sep",
    "oct":"oct","nov":"nov","dec":"dec"
  };

  const categories = {
    marital: ["single","married","divorced"],
    education: ["primary","secondary","tertiary","unknown"],
    job: ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
    contact: ["cellular","telephone"],
    poutcome: ["unknown","failure","other","success"]
  };

  function pickFromList(text, list) {
    const t = text.toLowerCase();
    for (const v of list) if (t.includes(v)) return v;
    const s = t.replace(/[^a-z]/g,"");
    for (const v of list) {
      const vs = v.replace(/[^a-z]/g,"");
      if (s.includes(vs)) return v;
    }
    return null;
  }

  function parseNumber(text) {
    const m = text.replace(/[,]/g,"").match(/-?\\d+(?:\\.\\d+)?/);
    return m ? Number(m[0]) : null;
  }

  function clampInt(v, lo, hi) {
    if (v == null || isNaN(v)) return null;
    v = Math.round(v);
    if (v < lo || v > hi) return null;
    return v;
  }

  function normMonth(text) {
    const words = text.toLowerCase().split(/\\s+/);
    for (const w of words) if (monthsMap[w]) return monthsMap[w];
    return null;
  }

  // --- Questions ---
  const fields = [
    { key:"age", prompt:"What is your age?", validate:(t)=>{ const n=parseNumber(t); return clampInt(n,10,120); }, confirm:(v)=>`Age ${v}.` },
    { key:"marital", prompt:"What is your marital status? Say single, married, or divorced.", validate:(t)=> pickFromList(t, categories.marital), confirm:(v)=>`Marital status ${v}.` },
    { key:"job", prompt:"What is your job?", validate:(t)=> pickFromList(t, categories.job), confirm:(v)=>`Job ${v}.` },
    { key:"education", prompt:"What is your education? Say primary, secondary, tertiary, or unknown.", validate:(t)=> pickFromList(t, categories.education), confirm:(v)=>`Education ${v}.` },
    { key:"default", prompt:"Do you have credit in default? Yes or no.", validate:(t)=> yn(t), confirm:(v)=>`Default ${v}.` },
    { key:"balance", prompt:"What is your current account balance? Say a number.", validate:(t)=> parseNumber(t), confirm:(v)=>`Balance ${v}.` },
    { key:"housing", prompt:"Do you have a housing loan? Yes or no.", validate:(t)=> yn(t), confirm:(v)=>`Housing loan ${v}.` },
    { key:"loan", prompt:"Do you have a personal loan? Yes or no.", validate:(t)=> yn(t), confirm:(v)=>`Personal loan ${v}.` },
    { key:"contact", prompt:"Preferred contact method: cellular or telephone?", validate:(t)=> pickFromList(t, categories.contact), confirm:(v)=>`Contact ${v}.` },
    { key:"day", prompt:"On which day of the month were you last contacted? Number 1 to 31.", validate:(t)=> clampInt(parseNumber(t),1,31), confirm:(v)=>`Day ${v}.` },
    { key:"month", prompt:"Say the last contact month.", validate:(t)=> normMonth(t), confirm:(v)=>`Month ${v}.` },
    { key:"duration", prompt:"Last contact duration in seconds?", validate:(t)=> clampInt(parseNumber(t),0,10000), confirm:(v)=>`Duration ${v} seconds.` },
    { key:"campaign", prompt:"How many contacts during this campaign?", validate:(t)=> clampInt(parseNumber(t),0,100), confirm:(v)=>`Campaign ${v}.` },
    { key:"pdays", prompt:"Days after last contact from a previous campaign?", validate:(t)=> parseNumber(t), confirm:(v)=>`Pdays ${v}.` },
    { key:"previous", prompt:"How many contacts before this campaign?", validate:(t)=> clampInt(parseNumber(t),0,100), confirm:(v)=>`Previous ${v}.` },
    { key:"poutcome", prompt:"Outcome of the previous marketing campaign?", validate:(t)=> pickFromList(t, categories.poutcome), confirm:(v)=>`Previous outcome ${v}.` },
  ];

  async function askAndGet(field) {
    for (let tries=0; tries<3; tries++) {
      status.textContent = field.prompt;
      await say(field.prompt);
      try {
        const heard = await listenOnce();
        addLog("Heard: " + heard);
        const val = field.validate(heard);
        if (val!==null && val!==undefined && val!=="") {
          const conf = field.confirm(val);
          status.textContent = conf;
          await say(conf);
          return val;
        } else {
          await say("Sorry, I didn't get that.");
        }
      } catch (e) {
        addLog("Recognition error: " + e.message);
        await say("Please repeat.");
      }
    }
    return null;
  }

  async function flow() {
    status.textContent = "Starting voice form…";
    await say("Let's fill your details by voice.");

    const answers = {};
    for (const f of fields) {
      const v = await askAndGet(f);
      answers[f.key] = v;
    }

    const lines = Object.entries(answers).map(([k,v]) => k + ": " + v);
    summary.style.display = "block";
    summary.innerHTML = "<b>Collected:</b><br>" + lines.join("<br>");
    status.textContent = "Review your details.";
    await say("Here is what I captured. Please review on screen.");
  }

  %s
})();
</script>
""" % ("flow();" if btn else "")

st.components.v1.html(html, height=520)
st.caption("Tip: Chrome works best. Allow microphone + sound.")
