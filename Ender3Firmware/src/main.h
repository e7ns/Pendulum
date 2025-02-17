// Macros for bit masks
#define TEST(n,b) (((n)&_BV(b))!=0)
#define SBI(n,b) (n |= _BV(b))
#define CBI(n,b) (n &= ~_BV(b))
#define SET_BIT(n,b,value) (n) ^= ((-value)^(n)) & (_BV(b))

void pciSetup(byte pin) {
  SBI(*digitalPinToPCMSK(pin), digitalPinToPCMSKbit(pin));  // enable pin
  SBI(PCIFR, digitalPinToPCICRbit(pin)); // clear any outstanding interrupt
  SBI(PCICR, digitalPinToPCICRbit(pin)); // enable interrupt for the group
}