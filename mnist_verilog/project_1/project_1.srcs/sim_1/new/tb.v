`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Testbench for BNN MNIST classifier - Tests all 10 digits
//////////////////////////////////////////////////////////////////////////////////

module tb;

    parameter integer CLK_HALF    = 5;      // 10ns period = 100MHz
    parameter integer TIMEOUT     = 2000;   // cycles per inference

    reg          clk;
    reg          rst_n;
    reg          start;
    reg  [783:0] image_in;

    reg  [3:0]   current_digit;
    wire [3:0]   digit_out;
    wire         valid;

    // Instantiate DUT
    top dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .image_in  (image_in),
        .digit_out (digit_out),
        .valid     (valid)
    );

    // Memory path - must match top.v
    localparam [8*10-1:0] MEM_DIR = "mem_files/";

    // Clock generation
    initial clk = 1'b0;
    always #(CLK_HALF) clk = ~clk;

    // Image buffer
    reg [783:0] img_buf [0:0];

    // Test results
    integer pass_count;
    integer fail_count;
    integer tc;
    integer popcount;
    integer i, d;

    // Task to load a specific digit image
    task load_image;
        input [3:0] digit;
        begin
            case (digit)
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
                    $display("ERROR: Invalid digit %0d", digit);
                    $finish;
                end
            endcase
            image_in = img_buf[0];

            // Count active pixels
            popcount = 0;
            for (i = 0; i < 784; i = i + 1) begin
                popcount = popcount + image_in[i];
            end
        end
    endtask

    // Task to run inference on one digit
    task run_inference;
        input [3:0] expected_digit;
        begin
            current_digit = expected_digit;
            load_image(expected_digit);

            $display("");
            $display("Testing digit %0d (active pixels: %0d)", expected_digit, popcount);

            // Send start pulse
            start = 1'b1;
            @(posedge clk);
            @(posedge clk);
            start = 1'b0;

            // Wait for valid
            tc = 0;
            while (!valid && tc < TIMEOUT) begin
                @(posedge clk);
                tc = tc + 1;
            end

            if (!valid) begin
                $display("  TIMEOUT after %0d cycles!", tc);
                fail_count = fail_count + 1;
            end else begin
                // Wait one more cycle for stable output
                @(posedge clk);

                // Check result
                if (digit_out === expected_digit) begin
                    $display("  PASS: Predicted %0d (correct) in %0d cycles", digit_out, tc);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  FAIL: Expected %0d, got %0d in %0d cycles", expected_digit, digit_out, tc);
                    fail_count = fail_count + 1;
                end

                // Show output scores
                $display("  Output scores:");
                for (i = 0; i < 10; i = i + 1) begin
                    if (i == digit_out)
                        $display("    Class %0d: %0d  <-- predicted", i, $signed(dut.out_acc[i]));
                    else
                        $display("    Class %0d: %0d", i, $signed(dut.out_acc[i]));
                end
            end

            // Wait a few cycles before next test
            repeat (5) @(posedge clk);
        end
    endtask

    // Main test sequence
    initial begin
        // Waveform dump
        $dumpfile("bnn_test.vcd");
        $dumpvars(0, tb);

        // Initialize
        rst_n    = 1'b0;
        start    = 1'b0;
        image_in = 784'b0;
        pass_count = 0;
        fail_count = 0;

        $display("");
        $display("==============================================");
        $display("BNN MNIST Classifier - Full Test Suite");
        $display("==============================================");

        // Verify weight loading by checking first weight entry
        $display("");
        $display("Memory loading verification:");
        $display("  w1[0] popcount: %0d (should be ~392 = half of 784)", $countones(dut.w1[0]));
        $display("  w2[0] popcount: %0d (should be ~256 = half of 512)", $countones(dut.w2[0]));
        $display("  thresh1[0]: %0d", dut.thresh1[0]);
        $display("  thresh2[0]: %0d", dut.thresh2[0]);
        $display("  invert1[0]: %0d", dut.invert1[0]);
        $display("  invert2[0]: %0d", dut.invert2[0]);
        $display("  b_out[0]: %0d", $signed(dut.b_out[0]));

        // Release reset
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        $display("");
        $display("Reset released, starting inference tests...");
        repeat (3) @(posedge clk);

        // Test all 10 digits
        for (d = 0; d < 10; d = d + 1) begin
            run_inference(d[3:0]);
        end

        // Summary
        $display("");
        $display("==============================================");
        $display("Test Summary:");
        $display("  Passed: %0d / 10", pass_count);
        $display("  Failed: %0d / 10", fail_count);
        $display("  Accuracy: %0d%%", pass_count * 10);
        $display("==============================================");

        if (pass_count >= 8) begin
            $display("OVERALL: SUCCESS (>=80%% accuracy)");
        end else begin
            $display("OVERALL: NEEDS IMPROVEMENT");
        end

        $display("");
        repeat (10) @(posedge clk);
        $finish;
    end

    // State monitor
    reg [2:0] prev_state;
    always @(posedge clk) begin
        if (rst_n && dut.state != prev_state) begin
            prev_state <= dut.state;
        end
    end

endmodule
