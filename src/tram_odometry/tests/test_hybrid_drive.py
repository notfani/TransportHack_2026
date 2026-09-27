"""Checks for ROS wiring of the delivered role-2 runtime."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tram_odometry.hybrid_drive import HybridDrive
CFG = Path(__file__).resolve().parents[1] / "config/hybrid.json"
def estimate(t, trust=1.0, slip=False):
    return SimpleNamespace(stamp_ns=t, trust_wheels=trust, slip_flag=slip)
def wheel(t, speed):
    return SimpleNamespace(stamp_ns=t, speed_mps=speed)
def test_trusted_history_and_slip():
    model = HybridDrive(CFG)
    base = 1_700_000_000_000_000_000
    for i in range(41):
        t = base+i*100_000_000
        model.predict_interval(5, 5.0+0.04*i, .05)
        model.observe_wheels(estimate(t), wheel(t, 5.0+0.04*i), wheel(t, 5.0+0.04*i))
        if i < 40:
            model.predict_interval(5, 5.0+0.04*i, .05)
            model.observe_wheels(estimate(t+50_000_000, trust=0), wheel(t, 5.0+0.04*i), wheel(t, 5.0+0.04*i))
    assert len(model.history)==11
    assert abs(model.state.last_accel-.4)<.001
    assert model.source=="hybrid"
    before=model.state.last_accel
    t=base+4_100_000_000
    model.predict_interval(5,6.64,.1)
    model.observe_wheels(estimate(t,trust=0,slip=True),wheel(t,30),wheel(t,30))
    assert not model.history
    assert model.state.last_accel==before
    assert model.state.trusted_age>0
def test_invalid_interval_resets_memory():
    model=HybridDrive(CFG)
    model.predict_interval(5,5,.1)
    drive=model.predict_interval(5,5,.6)
    assert drive.acceleration_mps2==0
    assert drive.sigma_mps2>=1
    assert model.source=="invalid_interval"
    assert model.state.warmup==0
