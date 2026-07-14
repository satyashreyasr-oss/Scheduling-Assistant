"""Quick manual check of tools.py logic (date resolution, booking, conflict
negotiation) without needing an LLM/API key. Run with: python smoke_test.py"""

import json
import os

# Use a throwaway DB so repeated runs don't collide with real bookings.
os.environ.setdefault("SMOKE_TEST", "1")
import tools

tools.DB_PATH = os.path.join(os.path.dirname(__file__), "data", "smoke_test.db")
if os.path.exists(tools.DB_PATH):
    os.remove(tools.DB_PATH)


def show(label, result):
    print(f"\n-- {label} --")
    print(json.dumps(json.loads(result), indent=2))


r = tools.resolve_date.invoke({"expression": "tomorrow"})
show("resolve_date('tomorrow')", r)
tomorrow = json.loads(r)["resolved_date"]

show("check_availability(tomorrow)", tools.check_availability.invoke({"date": tomorrow}))

show(
    "reserve_slot(tomorrow, 10:00)",
    tools.reserve_slot.invoke({"date": tomorrow, "time": "10:00", "email": "a@b.com"}),
)

show(
    "reserve_slot(tomorrow, 10:00) AGAIN -- should fail + suggest alternatives",
    tools.reserve_slot.invoke({"date": tomorrow, "time": "10:00", "email": "c@d.com"}),
)

show(
    "send_booking_notification (no webhook configured -> simulated)",
    tools.send_booking_notification.invoke({"email": "a@b.com", "details": f"Appt on {tomorrow} 10:00"}),
)

os.remove(tools.DB_PATH)
print("\nAll smoke tests ran without exceptions.")
