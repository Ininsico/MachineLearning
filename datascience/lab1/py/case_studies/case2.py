units = int(input("Enter units consumed (default 250): ").strip() or 250)

if units <= 100:
    rate = 10
elif units <= 200:
    rate = 15
elif units <= 300:
    rate = 20
else:
    rate = 25

bill = units * rate
service_charge = 500
total = bill + service_charge

print(f"Units Consumed: {units}")
print(f"Rate Applied: Rs. {rate}/unit")
print(f"Energy Bill: Rs. {bill:.2f}")
print(f"Service Charge: Rs. {service_charge:.2f}")
print(f"Total Payable: Rs. {total:.2f}")
