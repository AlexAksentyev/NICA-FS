clean:
	rm -v img/* tbl/*;
	rm ./output.txt;
	rm lyapunov.data;
	rm fort.*;

test: test.madx nica_24sol_rbend.seq
	(madx < $<) >output.txt
