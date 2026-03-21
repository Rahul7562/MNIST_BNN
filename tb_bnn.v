// =============================================================================
//  tb_bnn.v  –  Testbench for bnn_top
// =============================================================================
//  Loads a binarised test image from a .mem file, drives bnn_top,
//  and prints the predicted digit + waveform-visible digit_out bus.
//
//  Change TEST_DIGIT to 0–9 to test different images.
//  Simulated output for digit 2 should be:  digit_out = 4'b0010
//
//  Simulate with:
//    iverilog -o bnn_sim bnn_top.v tb_bnn.v && vvp bnn_sim
//  Or open in ModelSim / Vivado / Questa as usual.
// =============================================================================

`timescale 1ns / 1ps

module tb_bnn;

// ─────────────────────────────────────────────────────────────────────────────
//  Change this to test digit 0–9
// ─────────────────────────────────────────────────────────────────────────────
localparam integer TEST_DIGIT = 2;   // ← change me (expects output 0010 for 2)

// ─────────────────────────────────────────────────────────────────────────────
//  DUT signals
// ─────────────────────────────────────────────────────────────────────────────
reg          clk;
reg          rst_n;
reg          start;
reg  [783:0] image_in;
wire [3:0]   digit_out;
wire         valid;

// ─────────────────────────────────────────────────────────────────────────────
//  DUT instantiation
// ─────────────────────────────────────────────────────────────────────────────
bnn_top dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (start),
    .image_in  (image_in),
    .digit_out (digit_out),
    .valid     (valid)
);

// ─────────────────────────────────────────────────────────────────────────────
//  Clock: 10 ns period (100 MHz)
// ─────────────────────────────────────────────────────────────────────────────
initial clk = 0;
always #5 clk = ~clk;

// ─────────────────────────────────────────────────────────────────────────────
//  Load test image from .mem file
//  The file contains one line with 784 binary digits (one per pixel).
// ─────────────────────────────────────────────────────────────────────────────
reg [783:0] img_mem [0:0];   // 1-entry array of 784-bit values

// Build filename dynamically based on TEST_DIGIT
// (Verilog-2001 doesn't support string concat; use $sformat in SystemVerilog)
// For plain Verilog, we use a case statement to pick the file.
task load_image;
    begin
        case (TEST_DIGIT)
            0: $readmemb("mem_files/test_image_0.mem", img_mem);
            1: $readmemb("mem_files/test_image_1.mem", img_mem);
            2: $readmemb("mem_files/test_image_2.mem", img_mem);
            3: $readmemb("mem_files/test_image_3.mem", img_mem);
            4: $readmemb("mem_files/test_image_4.mem", img_mem);
            5: $readmemb("mem_files/test_image_5.mem", img_mem);
            6: $readmemb("mem_files/test_image_6.mem", img_mem);
            7: $readmemb("mem_files/test_image_7.mem", img_mem);
            8: $readmemb("mem_files/test_image_8.mem", img_mem);
            9: $readmemb("mem_files/test_image_9.mem", img_mem);
            default: begin
                $display("ERROR: TEST_DIGIT must be 0–9"); $finish;
            end
        endcase
        image_in = img_mem[0];
    end
endtask

// ─────────────────────────────────────────────────────────────────────────────
//  Stimulus
// ─────────────────────────────────────────────────────────────────────────────
integer timeout_cnt;

initial begin
    // Dump waveforms
    $dumpfile("bnn_wave.vcd");
    $dumpvars(0, tb_bnn);

    // Initialise
    rst_n = 0;
    start = 0;
    image_in = 784'b0;

    // Load image
    load_image();

    // Reset for 4 cycles
    @(posedge clk); #1;
    @(posedge clk); #1;
    rst_n = 1;
    @(posedge clk); #1;

    $display("");
    $display("============================================================");
    $display("  BNN Inference Testbench");
    $display("  Testing digit : %0d  (expected digit_out = %04b)", TEST_DIGIT, TEST_DIGIT[3:0]);
    $display("============================================================");

    // Pulse start for one cycle
    start = 1;
    @(posedge clk); #1;
    start = 0;

    // Wait for valid with timeout
    timeout_cnt = 0;
    while (!valid && timeout_cnt < 500) begin
        @(posedge clk); #1;
        timeout_cnt = timeout_cnt + 1;
    end

    if (timeout_cnt >= 500) begin
        $display("  ERROR: Timeout waiting for valid signal!");
        $finish;
    end

    // Sample output on the cycle valid is asserted
    @(posedge clk); #1;   // let digit_out settle with valid

    $display("");
    $display("  Input digit   : %0d", TEST_DIGIT);
    $display("  digit_out     : 4'b%04b  (%0d)", digit_out, digit_out);
    $display("  Expected      : 4'b%04b  (%0d)", TEST_DIGIT[3:0], TEST_DIGIT);

    if (digit_out === TEST_DIGIT[3:0])
        $display("  Result        : *** PASS ***");
    else
        $display("  Result        : *** FAIL ***  (got %0d, expected %0d)",
                  digit_out, TEST_DIGIT);

    $display("============================================================");
    $display("");

    // Run a few extra cycles to observe waveform
    repeat (10) @(posedge clk);

    $finish;
end

// ─────────────────────────────────────────────────────────────────────────────
//  Optional: continuous print whenever valid fires (useful for batch testing)
// ─────────────────────────────────────────────────────────────────────────────
always @(posedge clk) begin
    if (valid)
        $display("  [%0t ns]  valid=1  digit_out=4'b%04b  (%0d)",
                  $time, digit_out, digit_out);
end

endmodule
