const welcomeEl = document.getElementById('welcome');
const profilePicEl = document.getElementById('profile-pic');
const distanceSlider = document.getElementById('distance-slider');
const paceSlider = document.getElementById('pace-slider');
const distanceValue = document.getElementById('distance-value');
const paceValue = document.getElementById('pace-value');

const signinOverlay = document.getElementById('signin-overlay');

function getCookie(name) {
  const m = document.cookie.match(new RegExp('(?:^|; )' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '=([^;]*)'));
  return m ? decodeURIComponent(m[1]) : null;
}

(function () {
  try {
    const athleteCookie = getCookie('athlete');
    if (athleteCookie) {
      const payload = JSON.parse(atob(athleteCookie));
      if (payload.name) {
        welcomeEl.textContent = 'Hello, ' + payload.name;
        if (payload.profile_url) {
          profilePicEl.src = payload.profile_url;
          profilePicEl.classList.remove('hidden');
        }
        signinOverlay.classList.remove('visible');
        return;
      }
    }
  } catch (_) {}
  signinOverlay.classList.add('visible');
})();

distanceSlider.addEventListener('input', () => { distanceValue.textContent = distanceSlider.value; });
paceSlider.addEventListener('input', () => { paceValue.textContent = paceSlider.value; });

document.getElementById('predict-stress-btn').addEventListener('click', async () => {
  const resultEl = document.getElementById('predict-result');
  const distance = distanceSlider.value;
  const pace = paceSlider.value;
  resultEl.textContent = 'Loading…';
  try {
    const res = await fetch(`/predict_next_stress?distance=${distance}&pace=${pace}`, { credentials: 'include' });
    const data = await res.json();
    if (data.error) {
      resultEl.textContent = 'Error: ' + data.error;
    } else {
      resultEl.textContent = 'Predicted stress: ' + data.stress;
    }
  } catch (e) {
    resultEl.textContent = 'Request failed: ' + e.message;
  }
});
