`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:
// Engineer:
//
// Create Date: 20.03.2026 17:06:00
// Design Name:
// Module Name: tb
// Project Name:
// Target Devices:
// Tool Versions:
// Description: Testbench for BNN MNIST classifier
//
// Dependencies:
//
// Revision:
// Revision 0.02 - Fixed timing and added debugging
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////


module tb;

    parameter integer FOCUS_DIGIT = 2;
    parameter integer CLK_HALF    = 5;      // 10ns period
    parameter integer TIMEOUT     = 20000;  // must be > ~800 cycles

    reg          clk;
    reg          rst_n;
    reg          start;
    reg  [783:0] image_in;

    reg  [3:0]   image_id;
    wire [3:0]   digit_out;
    wire         valid;

    top dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .image_in  (image_in),
        .digit_out (digit_out),
        .valid     (valid)
    );

    // Keep this consistent with top.v
    localparam [8*10-1:0] MEM_DIR = "mem_files/";

    // Clock
    initial clk = 1'b0;
    always #(CLK_HALF) clk = ~clk;

    // Image loader
    reg [783:0] img_buf [0:0];

    task load_focus_image;
        begin
            case (FOCUS_DIGIT)
                0: $readmemb({MEM_DIR,"test_image_0.mem"}, img_buf);
                1: $readmemb({MEM_DIR,"test_image_1.mem"}, img_buf);
                2: $readmemb({MEM_DIR,"test_image_2.mem"}, img_buf);
                3: $readmemb({MEM_DIR,"test_image_3.mem"}, img_buf);
                4: $readmemb({MEM_DIR,"test_image_4.mem"}, img_buf);
                5: $readmemb({MEM_DIR,"test_image_5.mem"}, img_buf);
                6: $readmemb({MEM_DIR,"test_image_6.mem"}, img_buf);
                7: $readmemb({MEM_DIR,"test_image_7.mem"}, img_buf);
                8: $readmemb({MEM_DIR,"test_image_8.mem"}, img_buf);
                9: $readmemb({MEM_DIR,"test_image_9.mem"}, img_buf);
                default: begin
                    $display("ERROR: FOCUS_DIGIT must be 0..9");
                    $finish;
                end
            endcase
            image_in = img_buf[0];

            // Verify image was loaded (check if not all zeros/ones)
            if (image_in === 784'b0) begin
                $display("WARNING: Loaded image is all zeros - check mem_files path");
            end
            if (image_in === {784{1'b1}}) begin
                $display("WARNING: Loaded image is all ones - check mem_files path");
            end
            if (image_in === 784'bx) begin
                $display("ERROR: Image failed to load (all X) - check mem_files path");
            end
        end
    endtask

    integer tc;
    integer popcount;
    integer i;

    initial begin
        // Dump all signals for comprehensive waveform analysis
        $dumpfile("bnn_minimal.vcd");
        $dumpvars(0, tb);  // Dump entire testbench hierarchy including DUT

        // Display simulation start
        $display("===========================================");
        $display("BNN MNIST Classifier Simulation");
        $display("Testing digit: %0d", FOCUS_DIGIT);
        $display("===========================================");

        // init
        rst_n    = 1'b0;
        start    = 1'b0;
        image_in = 784'b0;
        image_id = FOCUS_DIGIT[3:0];

        // Load image
        load_focus_image();

        // Count active pixels in loaded image
        popcount = 0;
        for (i = 0; i < 784; i = i + 1) begin
            popcount = popcount + image_in[i];
        end
        $display("Image loaded: %0d active pixels out of 784", popcount);

        // release reset (sync) - wait for stable reset
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        $display("Reset released at time %0t", $time);

        // Wait for reset to propagate
        repeat (2) @(posedge clk);

        // Assert start pulse
        $display("Starting inference at time %0t", $time);
        start = 1'b1;
        repeat (3) @(posedge clk);
        start = 1'b0;

        // wait for valid with progress updates
        tc = 0;
        while (!valid && tc < TIMEOUT) begin
            @(posedge clk);
            tc = tc + 1;

            // Progress update every 200 cycles
            if (tc % 200 == 0) begin
                $display("  Processing... cycle %0d, state=%0d", tc, dut.state);
            end
        end

        if (!valid) begin
            $display("TIMEOUT waiting for valid after %0d cycles", tc);
            $display("Final state: %0d", dut.state);
            $finish;
        end

        // Report timing
        $display("Inference completed in %0d cycles", tc);
        $display("-------------------------------------------");

        // Wait one more cycle to ensure digit_out is stable
        @(posedge clk);

        // report results
        $display("Expected digit (image_id) = %0d", image_id);
        $display("Predicted digit_out       = %0d", digit_out);
        $display("-------------------------------------------");

        // Show output layer scores for debugging
        $display("Output layer scores:");
        for (i = 0; i < 10; i = i + 1) begin
            $display("  Class %0d: %0d", i, $signed(dut.out_acc[i]));
        end
        $display("-------------------------------------------");

        if (digit_out === image_id)
            $display("RESULT: PASS");
        else
            $display("RESULT: FAIL");

        repeat (10) @(posedge clk);
        $finish;
    end

    // Monitor state transitions for debugging
    reg [2:0] prev_state;
    always @(posedge clk) begin
        if (rst_n && dut.state != prev_state) begin
            case (dut.state)
                0: $display("  [%0t] State -> IDLE", $time);
                1: $display("  [%0t] State -> L1 (processing layer 1)", $time);
                2: $display("  [%0t] State -> L2 (processing layer 2)", $time);
                3: $display("  [%0t] State -> OUT (computing output layer)", $time);
                4: $display("  [%0t] State -> ARGMAX (finding max)", $time);
                5: $display("  [%0t] State -> DONE", $time);
            endcase
            prev_state <= dut.state;
        end
    end

endmodule
