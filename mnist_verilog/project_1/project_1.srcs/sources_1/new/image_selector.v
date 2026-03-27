`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Image Selector - Switch-based test image selection
//
// Stores 40 pre-loaded MNIST test images (4 per digit, 0-9)
// Selection via 6-bit switch input (0-39)
//
// Switch encoding:
//   sw[5:4] = variant (0-3): which of the 4 images for this digit
//   sw[3:0] = digit (0-9):   which digit
//
// Example: sw = 6'b01_0111 = variant 1 of digit 7
//
// Compatible with: Xilinx Vivado 2023.2, Zynq-7000
//////////////////////////////////////////////////////////////////////////////////

module image_selector (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [5:0]   sw,             // 6-bit switch input
    input  wire         load,           // Load selected image
    output reg  [783:0] image_out,      // 784-bit binary image
    output reg  [3:0]   expected_digit, // Expected digit for this image
    output reg          ready           // Image loaded and ready
);

    // -------------------------------------------------------------------------
    // Image Storage (40 images x 784 bits each)
    // -------------------------------------------------------------------------
    // Organized as: images[digit * 4 + variant]
    (* ram_style = "block" *) reg [783:0] images [0:39];

    // Memory path
    localparam MEM_DIR = "mem_files/";

    // -------------------------------------------------------------------------
    // Memory Initialization - Load all 40 test images
    // -------------------------------------------------------------------------
    initial begin
        // Digit 0, variants 0-3
        $readmemb({MEM_DIR, "test_image_0_0.mem"}, images, 0, 0);
        $readmemb({MEM_DIR, "test_image_0_1.mem"}, images, 1, 1);
        $readmemb({MEM_DIR, "test_image_0_2.mem"}, images, 2, 2);
        $readmemb({MEM_DIR, "test_image_0_3.mem"}, images, 3, 3);

        // Digit 1, variants 0-3
        $readmemb({MEM_DIR, "test_image_1_0.mem"}, images, 4, 4);
        $readmemb({MEM_DIR, "test_image_1_1.mem"}, images, 5, 5);
        $readmemb({MEM_DIR, "test_image_1_2.mem"}, images, 6, 6);
        $readmemb({MEM_DIR, "test_image_1_3.mem"}, images, 7, 7);

        // Digit 2, variants 0-3
        $readmemb({MEM_DIR, "test_image_2_0.mem"}, images, 8, 8);
        $readmemb({MEM_DIR, "test_image_2_1.mem"}, images, 9, 9);
        $readmemb({MEM_DIR, "test_image_2_2.mem"}, images, 10, 10);
        $readmemb({MEM_DIR, "test_image_2_3.mem"}, images, 11, 11);

        // Digit 3, variants 0-3
        $readmemb({MEM_DIR, "test_image_3_0.mem"}, images, 12, 12);
        $readmemb({MEM_DIR, "test_image_3_1.mem"}, images, 13, 13);
        $readmemb({MEM_DIR, "test_image_3_2.mem"}, images, 14, 14);
        $readmemb({MEM_DIR, "test_image_3_3.mem"}, images, 15, 15);

        // Digit 4, variants 0-3
        $readmemb({MEM_DIR, "test_image_4_0.mem"}, images, 16, 16);
        $readmemb({MEM_DIR, "test_image_4_1.mem"}, images, 17, 17);
        $readmemb({MEM_DIR, "test_image_4_2.mem"}, images, 18, 18);
        $readmemb({MEM_DIR, "test_image_4_3.mem"}, images, 19, 19);

        // Digit 5, variants 0-3
        $readmemb({MEM_DIR, "test_image_5_0.mem"}, images, 20, 20);
        $readmemb({MEM_DIR, "test_image_5_1.mem"}, images, 21, 21);
        $readmemb({MEM_DIR, "test_image_5_2.mem"}, images, 22, 22);
        $readmemb({MEM_DIR, "test_image_5_3.mem"}, images, 23, 23);

        // Digit 6, variants 0-3
        $readmemb({MEM_DIR, "test_image_6_0.mem"}, images, 24, 24);
        $readmemb({MEM_DIR, "test_image_6_1.mem"}, images, 25, 25);
        $readmemb({MEM_DIR, "test_image_6_2.mem"}, images, 26, 26);
        $readmemb({MEM_DIR, "test_image_6_3.mem"}, images, 27, 27);

        // Digit 7, variants 0-3
        $readmemb({MEM_DIR, "test_image_7_0.mem"}, images, 28, 28);
        $readmemb({MEM_DIR, "test_image_7_1.mem"}, images, 29, 29);
        $readmemb({MEM_DIR, "test_image_7_2.mem"}, images, 30, 30);
        $readmemb({MEM_DIR, "test_image_7_3.mem"}, images, 31, 31);

        // Digit 8, variants 0-3
        $readmemb({MEM_DIR, "test_image_8_0.mem"}, images, 32, 32);
        $readmemb({MEM_DIR, "test_image_8_1.mem"}, images, 33, 33);
        $readmemb({MEM_DIR, "test_image_8_2.mem"}, images, 34, 34);
        $readmemb({MEM_DIR, "test_image_8_3.mem"}, images, 35, 35);

        // Digit 9, variants 0-3
        $readmemb({MEM_DIR, "test_image_9_0.mem"}, images, 36, 36);
        $readmemb({MEM_DIR, "test_image_9_1.mem"}, images, 37, 37);
        $readmemb({MEM_DIR, "test_image_9_2.mem"}, images, 38, 38);
        $readmemb({MEM_DIR, "test_image_9_3.mem"}, images, 39, 39);
    end

    // -------------------------------------------------------------------------
    // Switch Decoding
    // -------------------------------------------------------------------------
    wire [1:0] variant = sw[5:4];  // 0-3
    wire [3:0] digit   = sw[3:0];  // 0-9

    // Calculate image index
    wire [5:0] img_idx = (digit < 4'd10) ? (digit * 4 + variant) : 6'd0;

    // -------------------------------------------------------------------------
    // Image Loading Logic
    // -------------------------------------------------------------------------
    reg load_d;  // Delayed load for edge detection

    always @(posedge clk) begin
        if (!rst_n) begin
            image_out      <= 784'b0;
            expected_digit <= 4'd0;
            ready          <= 1'b0;
            load_d         <= 1'b0;
        end else begin
            load_d <= load;

            // On rising edge of load
            if (load && !load_d) begin
                ready <= 1'b0;
            end

            // Load image on next cycle
            if (load_d && !ready) begin
                if (digit < 4'd10) begin
                    image_out      <= images[img_idx];
                    expected_digit <= digit;
                    ready          <= 1'b1;
                end else begin
                    // Invalid digit, output zeros
                    image_out      <= 784'b0;
                    expected_digit <= 4'd0;
                    ready          <= 1'b1;
                end
            end
        end
    end

endmodule
