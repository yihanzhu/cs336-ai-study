"""Unified rebuild. Requires the locally acquired source/evidence caches."""
import pathlib,subprocess,sys
W=pathlib.Path(__file__).parent;R=W.parent
for name in ['check_trace_values.py','build_audit_ledgers.py','build_audio_ledger.py','write_release_status.py','build_guide.py','build_materials.py','build_bilingual.py','verify_lesson_plans.py','verify_bilingual.py','verify_materials.py','verify_publication.py']:
 subprocess.run([sys.executable,str(W/name)],cwd=R,check=True)
