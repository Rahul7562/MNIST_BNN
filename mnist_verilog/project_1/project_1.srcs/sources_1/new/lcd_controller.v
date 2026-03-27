`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// LCD Controller for ST7735 1.8" TFT Display (128x160, SPI)
//
// Displays:
//   - 28x28 MNIST input image (scaled 3x to 84x84)
//   - Predicted digit (large font display)
//
// SPI Mode: Mode 0 (CPOL=0, CPHA=0)
// SPI Clock: ~8MHz (from 100MHz system clock, div=12)
//
// Compatible with: Xilinx Vivado 2023.2, Zynq-7000
//////////////////////////////////////////////////////////////////////////////////

module lcd_controller (
    input  wire         clk,            // 100MHz system clock
    input  wire         rst_n,
    input  wire         update,         // Trigger display update
    input  wire [783:0] image_in,       // 28x28 binary image
    input  wire [3:0]   digit_in,       // Predicted digit (0-9)
    input  wire [3:0]   expected_in,    // Expected digit (0-9)
    output reg          busy,           // Display update in progress

    // SPI interface
    output reg          lcd_cs_n,       // Chip select (active low)
    output reg          lcd_dc,         // Data/Command (0=cmd, 1=data)
    output reg          lcd_sclk,       // SPI clock
    output reg          lcd_mosi,       // SPI data out
    output reg          lcd_rst_n       // Display reset (active low)
);

    // -------------------------------------------------------------------------
    // Display Parameters
    // -------------------------------------------------------------------------
    localparam LCD_WIDTH  = 128;
    localparam LCD_HEIGHT = 160;
    localparam IMG_SIZE   = 28;
    localparam SCALE      = 3;           // Scale factor for image display
    localparam SCALED_SIZE = IMG_SIZE * SCALE;  // 84 pixels

    // Colors (RGB565 format)
    localparam [15:0] COLOR_WHITE  = 16'hFFFF;
    localparam [15:0] COLOR_BLACK  = 16'h0000;
    localparam [15:0] COLOR_GREEN  = 16'h07E0;
    localparam [15:0] COLOR_RED    = 16'hF800;
    localparam [15:0] COLOR_BLUE   = 16'h001F;
    localparam [15:0] COLOR_YELLOW = 16'hFFE0;

    // SPI clock divider (100MHz / 12 = 8.3MHz)
    localparam SPI_DIV = 6;  // Half period, so actual div = 12

    // -------------------------------------------------------------------------
    // State Machine
    // -------------------------------------------------------------------------
    localparam [3:0]
        ST_IDLE      = 4'd0,
        ST_RESET     = 4'd1,
        ST_INIT      = 4'd2,
        ST_SET_WINDOW = 4'd3,
        ST_DRAW_IMAGE = 4'd4,
        ST_DRAW_DIGIT = 4'd5,
        ST_DONE      = 4'd6;

    reg [3:0] state;
    reg [3:0] next_state;

    // -------------------------------------------------------------------------
    // Internal Registers
    // -------------------------------------------------------------------------
    reg [31:0] delay_cnt;
    reg [15:0] pixel_cnt;
    reg [7:0]  init_idx;
    reg [7:0]  spi_data;
    reg [3:0]  spi_bit_cnt;
    reg [7:0]  spi_clk_cnt;
    reg        spi_busy;
    reg        spi_is_data;  // 1=data, 0=command

    // Image coordinates
    reg [7:0] img_x, img_y;
    reg [15:0] pixel_color;

    // Digit font coordinates
    reg [6:0] font_x, font_y;
    reg [3:0] font_row;

    // Update request latch
    reg update_pending;
    reg init_done;

    // -------------------------------------------------------------------------
    // 7-Segment Style Digit Font (5x7 pixels, stored as 7 rows of 5 bits)
    // Large display: each pixel scaled 8x for visibility
    // -------------------------------------------------------------------------
    reg [4:0] digit_font [0:69];  // 10 digits x 7 rows

    initial begin
        // Digit 0
        digit_font[0]  = 5'b01110;
        digit_font[1]  = 5'b10001;
        digit_font[2]  = 5'b10011;
        digit_font[3]  = 5'b10101;
        digit_font[4]  = 5'b11001;
        digit_font[5]  = 5'b10001;
        digit_font[6]  = 5'b01110;

        // Digit 1
        digit_font[7]  = 5'b00100;
        digit_font[8]  = 5'b01100;
        digit_font[9]  = 5'b00100;
        digit_font[10] = 5'b00100;
        digit_font[11] = 5'b00100;
        digit_font[12] = 5'b00100;
        digit_font[13] = 5'b01110;

        // Digit 2
        digit_font[14] = 5'b01110;
        digit_font[15] = 5'b10001;
        digit_font[16] = 5'b00001;
        digit_font[17] = 5'b00110;
        digit_font[18] = 5'b01000;
        digit_font[19] = 5'b10000;
        digit_font[20] = 5'b11111;

        // Digit 3
        digit_font[21] = 5'b01110;
        digit_font[22] = 5'b10001;
        digit_font[23] = 5'b00001;
        digit_font[24] = 5'b00110;
        digit_font[25] = 5'b00001;
        digit_font[26] = 5'b10001;
        digit_font[27] = 5'b01110;

        // Digit 4
        digit_font[28] = 5'b00010;
        digit_font[29] = 5'b00110;
        digit_font[30] = 5'b01010;
        digit_font[31] = 5'b10010;
        digit_font[32] = 5'b11111;
        digit_font[33] = 5'b00010;
        digit_font[34] = 5'b00010;

        // Digit 5
        digit_font[35] = 5'b11111;
        digit_font[36] = 5'b10000;
        digit_font[37] = 5'b11110;
        digit_font[38] = 5'b00001;
        digit_font[39] = 5'b00001;
        digit_font[40] = 5'b10001;
        digit_font[41] = 5'b01110;

        // Digit 6
        digit_font[42] = 5'b00110;
        digit_font[43] = 5'b01000;
        digit_font[44] = 5'b10000;
        digit_font[45] = 5'b11110;
        digit_font[46] = 5'b10001;
        digit_font[47] = 5'b10001;
        digit_font[48] = 5'b01110;

        // Digit 7
        digit_font[49] = 5'b11111;
        digit_font[50] = 5'b00001;
        digit_font[51] = 5'b00010;
        digit_font[52] = 5'b00100;
        digit_font[53] = 5'b01000;
        digit_font[54] = 5'b01000;
        digit_font[55] = 5'b01000;

        // Digit 8
        digit_font[56] = 5'b01110;
        digit_font[57] = 5'b10001;
        digit_font[58] = 5'b10001;
        digit_font[59] = 5'b01110;
        digit_font[60] = 5'b10001;
        digit_font[61] = 5'b10001;
        digit_font[62] = 5'b01110;

        // Digit 9
        digit_font[63] = 5'b01110;
        digit_font[64] = 5'b10001;
        digit_font[65] = 5'b10001;
        digit_font[66] = 5'b01111;
        digit_font[67] = 5'b00001;
        digit_font[68] = 5'b00010;
        digit_font[69] = 5'b01100;
    end

    // -------------------------------------------------------------------------
    // ST7735 Initialization Commands
    // -------------------------------------------------------------------------
    reg [8:0] init_cmds [0:31];  // {DC, DATA[7:0]}

    initial begin
        // Software reset
        init_cmds[0]  = {1'b0, 8'h01};  // SWRESET
        init_cmds[1]  = {1'b0, 8'hFF};  // Delay marker

        // Sleep out
        init_cmds[2]  = {1'b0, 8'h11};  // SLPOUT
        init_cmds[3]  = {1'b0, 8'hFF};  // Delay marker

        // Color mode: 16-bit
        init_cmds[4]  = {1'b0, 8'h3A};  // COLMOD
        init_cmds[5]  = {1'b1, 8'h05};  // 16-bit color

        // Memory data access control
        init_cmds[6]  = {1'b0, 8'h36};  // MADCTL
        init_cmds[7]  = {1'b1, 8'h00};  // Normal orientation

        // Display on
        init_cmds[8]  = {1'b0, 8'h29};  // DISPON
        init_cmds[9]  = {1'b0, 8'hFF};  // Delay marker

        // End marker
        init_cmds[10] = {1'b0, 8'h00};
    end

    // -------------------------------------------------------------------------
    // SPI Transfer State Machine
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            lcd_sclk    <= 1'b0;
            lcd_mosi    <= 1'b0;
            spi_busy    <= 1'b0;
            spi_bit_cnt <= 4'd0;
            spi_clk_cnt <= 8'd0;
        end else if (spi_busy) begin
            if (spi_clk_cnt < SPI_DIV) begin
                spi_clk_cnt <= spi_clk_cnt + 8'd1;
            end else begin
                spi_clk_cnt <= 8'd0;
                lcd_sclk <= ~lcd_sclk;

                if (lcd_sclk) begin  // Falling edge - prepare next bit
                    if (spi_bit_cnt < 4'd8) begin
                        lcd_mosi <= spi_data[7 - spi_bit_cnt];
                        spi_bit_cnt <= spi_bit_cnt + 4'd1;
                    end else begin
                        spi_busy <= 1'b0;
                    end
                end
            end
        end
    end

    // Start SPI transfer task
    task start_spi;
        input [7:0] data;
        input is_data;
        begin
            spi_data    <= data;
            spi_is_data <= is_data;
            lcd_dc      <= is_data;
            spi_bit_cnt <= 4'd0;
            spi_clk_cnt <= 8'd0;
            spi_busy    <= 1'b1;
            lcd_mosi    <= data[7];
        end
    endtask

    // -------------------------------------------------------------------------
    // Get pixel from 28x28 image (with scaling)
    // -------------------------------------------------------------------------
    function get_image_pixel;
        input [7:0] x, y;
        input [783:0] img;
        reg [4:0] src_x, src_y;
        reg [9:0] src_idx;
        begin
            // Scale down coordinates
            src_x = x / SCALE;
            src_y = y / SCALE;

            if (src_x < IMG_SIZE && src_y < IMG_SIZE) begin
                src_idx = src_y * IMG_SIZE + src_x;
                get_image_pixel = img[src_idx];
            end else begin
                get_image_pixel = 1'b0;
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // Main Control State Machine
    // -------------------------------------------------------------------------
    reg [783:0] image_latch;
    reg [3:0]   digit_latch;
    reg [3:0]   expected_latch;
    reg [15:0]  draw_x, draw_y;

    // Font calculation variables (moved outside always block)
    reg [6:0]  rel_x, rel_y;
    reg [6:0]  font_idx;
    reg        font_pixel;

    always @(posedge clk) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            lcd_cs_n       <= 1'b1;
            lcd_rst_n      <= 1'b0;
            busy           <= 1'b0;
            delay_cnt      <= 32'd0;
            init_idx       <= 8'd0;
            init_done      <= 1'b0;
            update_pending <= 1'b0;
            img_x          <= 8'd0;
            img_y          <= 8'd0;
            pixel_cnt      <= 16'd0;
        end else begin

            // Latch update request
            if (update)
                update_pending <= 1'b1;

            case (state)

                ST_IDLE: begin
                    busy <= 1'b0;
                    if (!init_done) begin
                        state     <= ST_RESET;
                        delay_cnt <= 32'd0;
                        busy      <= 1'b1;
                    end else if (update_pending) begin
                        update_pending <= 1'b0;
                        image_latch    <= image_in;
                        digit_latch    <= digit_in;
                        expected_latch <= expected_in;
                        state          <= ST_SET_WINDOW;
                        busy           <= 1'b1;
                    end
                end

                ST_RESET: begin
                    // Hold reset low for 10ms
                    if (delay_cnt < 32'd1_000_000) begin
                        lcd_rst_n <= 1'b0;
                        delay_cnt <= delay_cnt + 32'd1;
                    end else if (delay_cnt < 32'd2_000_000) begin
                        lcd_rst_n <= 1'b1;
                        delay_cnt <= delay_cnt + 32'd1;
                    end else begin
                        state    <= ST_INIT;
                        init_idx <= 8'd0;
                    end
                end

                ST_INIT: begin
                    if (!spi_busy) begin
                        if (init_cmds[init_idx] == 9'h000) begin
                            // End of init sequence
                            init_done <= 1'b1;
                            lcd_cs_n  <= 1'b1;
                            state     <= ST_IDLE;
                        end else if (init_cmds[init_idx] == 9'h0FF) begin
                            // Delay command
                            if (delay_cnt < 32'd15_000_000) begin  // 150ms delay
                                delay_cnt <= delay_cnt + 32'd1;
                            end else begin
                                delay_cnt <= 32'd0;
                                init_idx  <= init_idx + 8'd1;
                            end
                        end else begin
                            lcd_cs_n <= 1'b0;
                            start_spi(init_cmds[init_idx][7:0], init_cmds[init_idx][8]);
                            init_idx <= init_idx + 8'd1;
                        end
                    end
                end

                ST_SET_WINDOW: begin
                    // Set column address (0 to LCD_WIDTH-1)
                    if (!spi_busy) begin
                        case (pixel_cnt)
                            0: begin lcd_cs_n <= 1'b0; start_spi(8'h2A, 0); end  // CASET
                            1: start_spi(8'h00, 1);  // X start high
                            2: start_spi(8'h00, 1);  // X start low
                            3: start_spi(8'h00, 1);  // X end high
                            4: start_spi(LCD_WIDTH - 1, 1);  // X end low

                            5: start_spi(8'h2B, 0);  // RASET
                            6: start_spi(8'h00, 1);  // Y start high
                            7: start_spi(8'h00, 1);  // Y start low
                            8: start_spi(8'h00, 1);  // Y end high
                            9: start_spi(LCD_HEIGHT - 1, 1);  // Y end low

                            10: start_spi(8'h2C, 0);  // RAMWR

                            11: begin
                                state     <= ST_DRAW_IMAGE;
                                pixel_cnt <= 16'd0;
                                draw_x    <= 16'd0;
                                draw_y    <= 16'd0;
                            end
                        endcase
                        if (pixel_cnt < 11)
                            pixel_cnt <= pixel_cnt + 16'd1;
                    end
                end

                ST_DRAW_IMAGE: begin
                    // Draw entire screen
                    if (!spi_busy) begin
                        // Determine pixel color based on region
                        if (draw_y < SCALED_SIZE + 10 && draw_x >= 22 && draw_x < 22 + SCALED_SIZE) begin
                            // MNIST image region (offset to center)
                            if (draw_y >= 10) begin
                                if (get_image_pixel(draw_x - 22, draw_y - 10, image_latch))
                                    pixel_color <= COLOR_WHITE;
                                else
                                    pixel_color <= COLOR_BLACK;
                            end else begin
                                pixel_color <= COLOR_BLACK;
                            end
                        end else if (draw_y >= 100) begin
                            // Digit display region
                            state <= ST_DRAW_DIGIT;
                        end else begin
                            // Background
                            pixel_color <= COLOR_BLACK;
                        end

                        // Send pixel
                        if (draw_y < 100) begin
                            if (pixel_cnt[0] == 0) begin
                                start_spi(pixel_color[15:8], 1);
                            end else begin
                                start_spi(pixel_color[7:0], 1);
                                draw_x <= draw_x + 16'd1;
                                if (draw_x >= LCD_WIDTH - 1) begin
                                    draw_x <= 16'd0;
                                    draw_y <= draw_y + 16'd1;
                                end
                            end
                            pixel_cnt <= pixel_cnt + 16'd1;
                        end
                    end
                end

                ST_DRAW_DIGIT: begin
                    // Draw predicted digit with color coding
                    if (!spi_busy) begin
                        // Calculate font pixel
                        rel_x = draw_x;
                        rel_y = draw_y - 100;

                        // Digit position: centered, scaled 6x
                        if (rel_x >= 40 && rel_x < 40 + 30 && rel_y >= 20 && rel_y < 20 + 42) begin
                            font_idx = digit_latch * 7 + ((rel_y - 20) / 6);
                            font_pixel = digit_font[font_idx][4 - ((rel_x - 40) / 6)];

                            if (font_pixel) begin
                                if (digit_latch == expected_latch)
                                    pixel_color <= COLOR_GREEN;  // Correct
                                else
                                    pixel_color <= COLOR_RED;    // Wrong
                            end else begin
                                pixel_color <= COLOR_BLACK;
                            end
                        end else begin
                            pixel_color <= COLOR_BLACK;
                        end

                        // Send pixel
                        if (pixel_cnt[0] == 0) begin
                            start_spi(pixel_color[15:8], 1);
                        end else begin
                            start_spi(pixel_color[7:0], 1);
                            draw_x <= draw_x + 16'd1;
                            if (draw_x >= LCD_WIDTH - 1) begin
                                draw_x <= 16'd0;
                                draw_y <= draw_y + 16'd1;
                                if (draw_y >= LCD_HEIGHT) begin
                                    state <= ST_DONE;
                                end
                            end
                        end
                        pixel_cnt <= pixel_cnt + 16'd1;
                    end
                end

                ST_DONE: begin
                    lcd_cs_n  <= 1'b1;
                    busy      <= 1'b0;
                    pixel_cnt <= 16'd0;
                    state     <= ST_IDLE;
                end

            endcase
        end
    end

endmodule
