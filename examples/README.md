# Hardware examples

The blinky example builds a resettable counter through Vulcan's circuit API. It is an example
design, not a compiler frontend:

```sh
zig build run-blinky -- --width 24 -o blinky.v
zig build run-blinky -- --width 24 --format json -o blinky.json
```

The generated module has `clk`, `reset`, and `led` ports. Add board-specific pin constraints,
then run the Verilog or JSON through Yosys and nextpnr for the FPGA family and board in use.

The same design is available in the textual structural format used by `vulcan-irc`:

```sh
zig build vulcan-irc -- --target netlist examples/blinky.circuit -o blinky.v
zig build vulcan-irc -- --target netlist --format json examples/blinky.circuit -o blinky.json
```
