clean:
	rm -v img/* tbl/*;
	rm ./output*.txt;
	rm lyapunov.data;
	rm fort.*;

test24: test24.madx nica_24sol_rbend.seq
	(madx < $<) >output24.txt

testNICA: testNICA.madx NICA_full.seq
	(madx < $<) >outputNICA.txt

testSHIFT: testSHIFT.madx NICA_SOL_shifted_RBEND.seq
	(madx < $<) >outputSHIFT.txt
