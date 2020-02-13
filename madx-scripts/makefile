clean:
	rm -v img/* tbl/*;

test: test.madx nica_24sol_rbend.seq
	(madx < $<) >output.txt
