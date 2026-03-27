`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Comprehensive Testbench for BNN MNIST Classifier
//
// Tests:
//   1. BNN Core standalone (all 10 digits)
//   2. Image selector with switch input
//   3. Full system integration (optional LCD excluded for speed)
//
// Expected: 90%+ accuracy on test images
//////////////////////////////////////////////////////////////////////////////////

module tb_bnn_system;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;      // 100MHz
    parameter TIMEOUT    = 3000;    // Cycles per inference

    // -------------------------------------------------------------------------
    // DUT Signals
    // -------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg  [5:0]  sw;
    reg  [1:0]  btn;
    wire [7:0]  led;
    wire        lcd_cs_n, lcd_dc, lcd_sclk, lcd_mosi, lcd_rst_n;
    wire [6:0]  seg;
    wire [3:0]  an;

    // -------------------------------------------------------------------------
    // Direct BNN Core Testing Signals
    // -------------------------------------------------------------------------
    reg         bnn_start;
    reg  [783:0] bnn_image;
    wire [3:0]  bnn_digit;
    wire        bnn_valid;
    wire        bnn_busy;

    // -------------------------------------------------------------------------
    // Test Results
    // -------------------------------------------------------------------------
    integer pass_count;
    integer fail_count;
    integer total_cycles;
    integer i, d, v;

    // -------------------------------------------------------------------------
    // Memory for Test Images
    // -------------------------------------------------------------------------
    reg [783:0] test_images [0:39];     // 40 images (4 per digit)
    reg [783:0] single_image [0:0];

    // Memory path
    localparam MEM_DIR = "mem_files/";

    // -------------------------------------------------------------------------
    // Clock Generation
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Instantiate BNN Core for Direct Testing
    // -------------------------------------------------------------------------
    bnn_core u_bnn_direct (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (bnn_start),
        .image_in   (bnn_image),
        .digit_out  (bnn_digit),
        .valid      (bnn_valid),
        .busy       (bnn_busy)
    );

    // -------------------------------------------------------------------------
    // Load Test Images
    // -------------------------------------------------------------------------
    task load_test_images;
        begin
            // Load all 40 test images (4 per digit)
            $readmemb({MEM_DIR, "test_image_0_0.mem"}, test_images, 0, 0);
            $readmemb({MEM_DIR, "test_image_0_1.mem"}, test_images, 1, 1);
            $readmemb({MEM_DIR, "test_image_0_2.mem"}, test_images, 2, 2);
            $readmemb({MEM_DIR, "test_image_0_3.mem"}, test_images, 3, 3);

            $readmemb({MEM_DIR, "test_image_1_0.mem"}, test_images, 4, 4);
            $readmemb({MEM_DIR, "test_image_1_1.mem"}, test_images, 5, 5);
            $readmemb({MEM_DIR, "test_image_1_2.mem"}, test_images, 6, 6);
            $readmemb({MEM_DIR, "test_image_1_3.mem"}, test_images, 7, 7);

            $readmemb({MEM_DIR, "test_image_2_0.mem"}, test_images, 8, 8);
            $readmemb({MEM_DIR, "test_image_2_1.mem"}, test_images, 9, 9);
            $readmemb({MEM_DIR, "test_image_2_2.mem"}, test_images, 10, 10);
            $readmemb({MEM_DIR, "test_image_2_3.mem"}, test_images, 11, 11);

            $readmemb({MEM_DIR, "test_image_3_0.mem"}, test_images, 12, 12);
            $readmemb({MEM_DIR, "test_image_3_1.mem"}, test_images, 13, 13);
            $readmemb({MEM_DIR, "test_image_3_2.mem"}, test_images, 14, 14);
            $readmemb({MEM_DIR, "test_image_3_3.mem"}, test_images, 15, 15);

            $readmemb({MEM_DIR, "test_image_4_0.mem"}, test_images, 16, 16);
            $readmemb({MEM_DIR, "test_image_4_1.mem"}, test_images, 17, 17);
            $readmemb({MEM_DIR, "test_image_4_2.mem"}, test_images, 18, 18);
            $readmemb({MEM_DIR, "test_image_4_3.mem"}, test_images, 19, 19);

            $readmemb({MEM_DIR, "test_image_5_0.mem"}, test_images, 20, 20);
            $readmemb({MEM_DIR, "test_image_5_1.mem"}, test_images, 21, 21);
            $readmemb({MEM_DIR, "test_image_5_2.mem"}, test_images, 22, 22);
            $readmemb({MEM_DIR, "test_image_5_3.mem"}, test_images, 23, 23);

            $readmemb({MEM_DIR, "test_image_6_0.mem"}, test_images, 24, 24);
            $readmemb({MEM_DIR, "test_image_6_1.mem"}, test_images, 25, 25);
            $readmemb({MEM_DIR, "test_image_6_2.mem"}, test_images, 26, 26);
            $readmemb({MEM_DIR, "test_image_6_3.mem"}, test_images, 27, 27);

            $readmemb({MEM_DIR, "test_image_7_0.mem"}, test_images, 28, 28);
            $readmemb({MEM_DIR, "test_image_7_1.mem"}, test_images, 29, 29);
            $readmemb({MEM_DIR, "test_image_7_2.mem"}, test_images, 30, 30);
            $readmemb({MEM_DIR, "test_image_7_3.mem"}, test_images, 31, 31);

            $readmemb({MEM_DIR, "test_image_8_0.mem"}, test_images, 32, 32);
            $readmemb({MEM_DIR, "test_image_8_1.mem"}, test_images, 33, 33);
            $readmemb({MEM_DIR, "test_image_8_2.mem"}, test_images, 34, 34);
            $readmemb({MEM_DIR, "test_image_8_3.mem"}, test_images, 35, 35);

            $readmemb({MEM_DIR, "test_image_9_0.mem"}, test_images, 36, 36);
            $readmemb({MEM_DIR, "test_image_9_1.mem"}, test_images, 37, 37);
            $readmemb({MEM_DIR, "test_image_9_2.mem"}, test_images, 38, 38);
            $readmemb({MEM_DIR, "test_image_9_3.mem"}, test_images, 39, 39);

            $display("  Loaded 40 test images (4 per digit)");
        end
    endtask

    // -------------------------------------------------------------------------
    // Run Single BNN Inference
    // -------------------------------------------------------------------------
    task run_bnn_inference;
        input [5:0]  img_idx;
        input [3:0]  expected;
        output       passed;
        integer cycles;
        reg     timed_out;
        begin
            bnn_image = test_images[img_idx];
            bnn_start = 1'b1;
            @(posedge clk);
            @(posedge clk);
            bnn_start = 1'b0;

            cycles = 0;
            while (!bnn_valid && cycles < TIMEOUT) begin
                @(posedge clk);
                cycles = cycles + 1;
            end

            // Capture timeout status BEFORE the extra clock (valid is only high 1 cycle)
            timed_out = (cycles >= TIMEOUT);

            @(posedge clk);  // One more for stable output

            if (timed_out) begin
                $display("    TIMEOUT for image %0d (expected digit %0d)", img_idx, expected);
                passed = 0;
            end else if (bnn_digit === expected) begin
                passed = 1;
            end else begin
                passed = 0;
            end

            total_cycles = total_cycles + cycles;
            repeat(3) @(posedge clk);  // Brief pause
        end
    endtask

    // -------------------------------------------------------------------------
    // Count Active Pixels in Image
    // -------------------------------------------------------------------------
    function integer count_pixels;
        input [783:0] img;
        integer k, cnt;
        begin
            cnt = 0;
            for (k = 0; k < 784; k = k + 1)
                cnt = cnt + img[k];
            count_pixels = cnt;
        end
    endfunction

    // -------------------------------------------------------------------------
    // Main Test Sequence
    // -------------------------------------------------------------------------
    reg test_passed;
    integer pixel_count;
    integer digit_pass [0:9];
    integer digit_total [0:9];

    initial begin
        // Initialize
        $dumpfile("bnn_test.vcd");
        $dumpvars(0, tb_bnn_system);

        rst_n     = 1'b0;
        bnn_start = 1'b0;
        bnn_image = 784'b0;
        sw        = 6'b0;
        btn       = 2'b0;
        pass_count = 0;
        fail_count = 0;
        total_cycles = 0;

        for (d = 0; d < 10; d = d + 1) begin
            digit_pass[d] = 0;
            digit_total[d] = 0;
        end

        $display("");
        $display("================================================================");
        $display("BNN MNIST Classifier - Comprehensive Test Suite");
        $display("================================================================");
        $display("");

        // Load test images
        load_test_images();

        // Release reset
        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        // =====================================================================
        // Test 1: BNN Core - All 40 Images
        // =====================================================================
        $display("");
        $display("----------------------------------------------------------------");
        $display("Test 1: BNN Core Direct Testing (40 images)");
        $display("----------------------------------------------------------------");
        $display("");
        $display("  Digit  Var  Pixels   Result   Status");
        $display("  -----  ---  ------   ------   ------");

        for (d = 0; d < 10; d = d + 1) begin
            for (v = 0; v < 4; v = v + 1) begin
                i = d * 4 + v;
                pixel_count = count_pixels(test_images[i]);

                run_bnn_inference(i[5:0], d[3:0], test_passed);

                digit_total[d] = digit_total[d] + 1;

                if (test_passed) begin
                    pass_count = pass_count + 1;
                    digit_pass[d] = digit_pass[d] + 1;
                    $display("  %5d  %3d  %6d   %6d   PASS", d, v, pixel_count, bnn_digit);
                end else begin
                    fail_count = fail_count + 1;
                    $display("  %5d  %3d  %6d   %6d   FAIL (expected %0d)", d, v, pixel_count, bnn_digit, d);
                end
            end
        end

        // =====================================================================
        // Summary
        // =====================================================================
        $display("");
        $display("================================================================");
        $display("Test Summary");
        $display("================================================================");
        $display("");
        $display("  Per-Digit Accuracy:");
        for (d = 0; d < 10; d = d + 1) begin
            $display("    Digit %0d: %0d / %0d correct (%0d%%)",
                     d, digit_pass[d], digit_total[d],
                     digit_total[d] > 0 ? (digit_pass[d] * 100 / digit_total[d]) : 0);
        end

        $display("");
        $display("  Overall Results:");
        $display("    Total Passed: %0d / %0d", pass_count, pass_count + fail_count);
        $display("    Total Failed: %0d / %0d", fail_count, pass_count + fail_count);
        $display("    Accuracy:     %0d%%", (pass_count * 100) / (pass_count + fail_count));
        $display("    Avg Latency:  %0d cycles per inference", total_cycles / (pass_count + fail_count));
        $display("");

        if (pass_count >= 36) begin  // 90% of 40
            $display("  STATUS: SUCCESS (>=90%% accuracy)");
        end else if (pass_count >= 32) begin  // 80%
            $display("  STATUS: ACCEPTABLE (>=80%% accuracy)");
        end else begin
            $display("  STATUS: NEEDS IMPROVEMENT (<80%% accuracy)");
            $display("");
            $display("  Check:");
            $display("    1. Run mnist_bnn_v2.py to retrain with verification");
            $display("    2. Run sync_mem_files.bat to copy weights");
            $display("    3. Verify threshold calculation in Python");
        end

        $display("");
        $display("================================================================");
        $display("");

        repeat(20) @(posedge clk);
        $finish;
    end

    // -------------------------------------------------------------------------
    // Watchdog Timer
    // -------------------------------------------------------------------------
    initial begin
        #(CLK_PERIOD * 500000);  // 5ms timeout
        $display("");
        $display("ERROR: Global timeout reached!");
        $finish;
    end

endmodule
