import Std

/-!
Source-bound integer fragment of the active OKC MULDEC decoder.

The upstream SF Tensor paper motivates machine-checked kernel properties, but
its Lean PTX implementation was not publicly released at this review. This is
an independent fixed-width fragment, not an import of that model or a complete
PTX interpreter. tools/verify_ptx.py owns the separate source-correspondence
check. The specification extracts each code bit from the Python packing map;
it does not reuse the implementation's rotations or multiplication trick.

No theorem below interprets the resulting words as floating-point numbers or
claims correctness of scale multiplication, WGMMA, memory ordering or CUDA.
-/

namespace OKC.Dq4

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

abbrev Word := BitVec 32
abbrev Code := BitVec 4

def mask : Word := 0x81C081C0

-- The active unsigned32 C++ expressions. Identical funnel operands at the
-- fixed shifts 3 and 6 implement rotate-right; Word multiplication wraps.
def decode0 (q : Word) : Word := q &&& mask
def decode1 (q : Word) : Word := q.rotateRight 3 &&& mask
def decode2 (q : Word) : Word := q.rotateRight 6 &&& mask
def decode3 (q : Word) : Word := ((q &&& 0x001B001B) * 4228) &&& mask

-- Independent bit-placement specification: extract four code bits, then
-- expand into positions 6,7,8,15 of a half word. No floating-point operations.
def bit (q : Word) (p : Nat) : Word := (q >>> p) &&& 1

def extractCode (q : Word) (p0 p1 p2 p3 : Nat) : Code :=
  (bit q p0 ||| (bit q p1 <<< 1) ||| (bit q p2 <<< 2) |||
    (bit q p3 <<< 3)).setWidth 4

def expand (c : Code) : Word :=
  let w := c.setWidth 32
  ((w &&& 7) <<< 6) ||| ((w &&& 8) <<< 12)

def expandPair (lo hi : Code) : Word := expand lo ||| (expand hi <<< 16)

-- Register/half order is R0lo,R0hi,R1lo,R1hi,R2lo,R2hi,R3lo,R3hi.
-- R0..3 correspond to _pack_codes_nat's (h,q)=(0,0),(1,0),(0,1),(1,1).
def code00 (q : Word) : Code := extractCode q 6 7 8 15
def code01 (q : Word) : Code := extractCode q 22 23 24 31
def code10 (q : Word) : Code := extractCode q 9 10 11 18
def code11 (q : Word) : Code := extractCode q 25 26 27 2
def code20 (q : Word) : Code := extractCode q 12 13 14 21
def code21 (q : Word) : Code := extractCode q 28 29 30 5
def code30 (q : Word) : Code := extractCode q 4 0 1 3
def code31 (q : Word) : Code := extractCode q 20 16 17 19

def spec0 (q : Word) : Word := expandPair (code00 q) (code01 q)
def spec1 (q : Word) : Word := expandPair (code10 q) (code11 q)
def spec2 (q : Word) : Word := expandPair (code20 q) (code21 q)
def spec3 (q : Word) : Word := expandPair (code30 q) (code31 q)

theorem decode0_correct (q : Word) : decode0 q = spec0 q := by
  apply BitVec.eq_of_getLsbD_eq_iff.mpr
  intro i hi
  have hc : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 ∨ i = 8 ∨ i = 9 ∨ i = 10 ∨ i = 11 ∨ i = 12 ∨ i = 13 ∨ i = 14 ∨ i = 15 ∨ i = 16 ∨ i = 17 ∨ i = 18 ∨ i = 19 ∨ i = 20 ∨ i = 21 ∨ i = 22 ∨ i = 23 ∨ i = 24 ∨ i = 25 ∨ i = 26 ∨ i = 27 ∨ i = 28 ∨ i = 29 ∨ i = 30 ∨ i = 31 := by omega
  rcases hc with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst i <;>
    simp [decode0, spec0, code00, code01, expandPair, expand, extractCode, bit, mask]

theorem decode1_correct (q : Word) : decode1 q = spec1 q := by
  apply BitVec.eq_of_getLsbD_eq_iff.mpr
  intro i hi
  have hc : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 ∨ i = 8 ∨ i = 9 ∨ i = 10 ∨ i = 11 ∨ i = 12 ∨ i = 13 ∨ i = 14 ∨ i = 15 ∨ i = 16 ∨ i = 17 ∨ i = 18 ∨ i = 19 ∨ i = 20 ∨ i = 21 ∨ i = 22 ∨ i = 23 ∨ i = 24 ∨ i = 25 ∨ i = 26 ∨ i = 27 ∨ i = 28 ∨ i = 29 ∨ i = 30 ∨ i = 31 := by omega
  rcases hc with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst i <;>
    simp [decode1, spec1, code10, code11, expandPair, expand, extractCode, bit, mask]

theorem decode2_correct (q : Word) : decode2 q = spec2 q := by
  apply BitVec.eq_of_getLsbD_eq_iff.mpr
  intro i hi
  have hc : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 ∨ i = 8 ∨ i = 9 ∨ i = 10 ∨ i = 11 ∨ i = 12 ∨ i = 13 ∨ i = 14 ∨ i = 15 ∨ i = 16 ∨ i = 17 ∨ i = 18 ∨ i = 19 ∨ i = 20 ∨ i = 21 ∨ i = 22 ∨ i = 23 ∨ i = 24 ∨ i = 25 ∨ i = 26 ∨ i = 27 ∨ i = 28 ∨ i = 29 ∨ i = 30 ∨ i = 31 := by omega
  rcases hc with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst i <;>
    simp [decode2, spec2, code20, code21, expandPair, expand, extractCode, bit, mask]

def positions : List Nat :=
  [6,7,8,15,22,23,24,31,9,10,11,18,25,26,27,2,
   12,13,14,21,28,29,30,5,4,0,1,3,20,16,17,19]

-- Every stored bit is used exactly once. This is a kernel-checked finite
-- proposition, with no native-code evaluation trusted as a proof.
theorem positions_bijective : positions.Perm (List.range 32) := by decide

def placeCode (c : Code) (p0 p1 p2 p3 : Nat) : Word :=
  let w := c.setWidth 32
  (bit w 0 <<< p0) ||| (bit w 1 <<< p1) |||
    (bit w 2 <<< p2) ||| (bit w 3 <<< p3)

-- Only the eight masked input bits influence the multiply. This identity is
-- proved symbolically for every Word; the following finite lemma then covers
-- all 16*16 assignments to those bits, using kernel reduction only.
theorem masked3 (q : Word) :
    q &&& 0x001B001B =
      placeCode (code30 q) 4 0 1 3 ||| placeCode (code31 q) 20 16 17 19 := by
  apply BitVec.eq_of_getLsbD_eq_iff.mpr
  intro i hi
  have hc : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 ∨ i = 8 ∨ i = 9 ∨ i = 10 ∨ i = 11 ∨ i = 12 ∨ i = 13 ∨ i = 14 ∨ i = 15 ∨ i = 16 ∨ i = 17 ∨ i = 18 ∨ i = 19 ∨ i = 20 ∨ i = 21 ∨ i = 22 ∨ i = 23 ∨ i = 24 ∨ i = 25 ∨ i = 26 ∨ i = 27 ∨ i = 28 ∨ i = 29 ∨ i = 30 ∨ i = 31 := by omega
  rcases hc with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst i <;>
    simp [placeCode, code30, code31, extractCode, bit]

theorem multiply3_codes : ∀ (lo hi : Code),
    (((placeCode lo 4 0 1 3 ||| placeCode hi 20 16 17 19) * 4228) &&& mask) =
      expandPair lo hi := by decide

theorem decode3_correct (q : Word) : decode3 q = spec3 q := by
  unfold decode3 spec3
  rw [masked3]
  exact multiply3_codes (code30 q) (code31 q)

#print axioms masked3
#print axioms multiply3_codes
#print axioms decode3_correct

#print axioms decode0_correct
#print axioms decode1_correct
#print axioms decode2_correct
#print axioms positions_bijective
end OKC.Dq4
