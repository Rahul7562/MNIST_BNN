################################################################################
# BNN MNIST FPGA Constraints for Zynq-7000 (ZedBoard / PYNQ-Z1)
#
# Target: xc7z020clg484-1
# Clock: 100MHz from PL
#
# Modify pin assignments below to match your specific board revision.
# The default assignments are for ZedBoard.
################################################################################

################################################################################
# Clock Constraint
################################################################################
# System clock (100MHz PL clock on ZedBoard is Y9)
set_property PACKAGE_PIN Y9 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk]

################################################################################
# Reset Button (Active Low)
################################################################################
# BTN_CENTER on ZedBoard (directly to PL)
set_property PACKAGE_PIN P16 [get_ports rst_n]
set_property IOSTANDARD LVCMOS25 [get_ports rst_n]

################################################################################
# Slide Switches (directly on ZedBoard)
################################################################################
# SW0-SW5 for image selection
set_property PACKAGE_PIN F22 [get_ports {sw[0]}]
set_property PACKAGE_PIN G22 [get_ports {sw[1]}]
set_property PACKAGE_PIN H22 [get_ports {sw[2]}]
set_property PACKAGE_PIN F21 [get_ports {sw[3]}]
set_property PACKAGE_PIN H19 [get_ports {sw[4]}]
set_property PACKAGE_PIN H18 [get_ports {sw[5]}]

set_property IOSTANDARD LVCMOS25 [get_ports {sw[*]}]

################################################################################
# Push Buttons (directly on ZedBoard)
################################################################################
# BTN_LEFT  (directly to PL) - Run inference
# BTN_RIGHT (directly to PL) - Update LCD
set_property PACKAGE_PIN N15 [get_ports {btn[0]}]
set_property PACKAGE_PIN R18 [get_ports {btn[1]}]

set_property IOSTANDARD LVCMOS25 [get_ports {btn[*]}]

################################################################################
# LEDs (directly on ZedBoard)
################################################################################
# LD0-LD7 directly on PL
set_property PACKAGE_PIN T22 [get_ports {led[0]}]
set_property PACKAGE_PIN T21 [get_ports {led[1]}]
set_property PACKAGE_PIN U22 [get_ports {led[2]}]
set_property PACKAGE_PIN U21 [get_ports {led[3]}]
set_property PACKAGE_PIN V22 [get_ports {led[4]}]
set_property PACKAGE_PIN W22 [get_ports {led[5]}]
set_property PACKAGE_PIN U19 [get_ports {led[6]}]
set_property PACKAGE_PIN U14 [get_ports {led[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

################################################################################
# Pmod Connector JB (ST7735 LCD via SPI)
################################################################################
# JB1 - lcd_cs_n   (directly from PL)
# JB2 - lcd_dc
# JB3 - lcd_sclk
# JB4 - lcd_mosi
# JB7 - lcd_rst_n

# Pmod JB directly on PL
set_property PACKAGE_PIN W12 [get_ports lcd_cs_n]
set_property PACKAGE_PIN W11 [get_ports lcd_dc]
set_property PACKAGE_PIN V10 [get_ports lcd_sclk]
set_property PACKAGE_PIN W8  [get_ports lcd_mosi]
set_property PACKAGE_PIN V12 [get_ports lcd_rst_n]

set_property IOSTANDARD LVCMOS33 [get_ports lcd_cs_n]
set_property IOSTANDARD LVCMOS33 [get_ports lcd_dc]
set_property IOSTANDARD LVCMOS33 [get_ports lcd_sclk]
set_property IOSTANDARD LVCMOS33 [get_ports lcd_mosi]
set_property IOSTANDARD LVCMOS33 [get_ports lcd_rst_n]

# SPI timing constraints
set_property SLEW FAST [get_ports lcd_sclk]
set_property SLEW FAST [get_ports lcd_mosi]
set_property DRIVE 8 [get_ports lcd_sclk]
set_property DRIVE 8 [get_ports lcd_mosi]

################################################################################
# Pmod Connector JA (Seven Segment Display - Optional)
################################################################################
# If using a Pmod SSD or discrete seven-segment display
# JA1-JA4 = Segments a-d (directly from PL)
# JA7-JA10 = Segments e-g + Anode

set_property PACKAGE_PIN Y11  [get_ports {seg[0]}]
set_property PACKAGE_PIN AA11 [get_ports {seg[1]}]
set_property PACKAGE_PIN Y10  [get_ports {seg[2]}]
set_property PACKAGE_PIN AA9  [get_ports {seg[3]}]
set_property PACKAGE_PIN AB11 [get_ports {seg[4]}]
set_property PACKAGE_PIN AB10 [get_ports {seg[5]}]
set_property PACKAGE_PIN AB9  [get_ports {seg[6]}]

set_property PACKAGE_PIN AA8  [get_ports {an[0]}]
set_property PACKAGE_PIN AB8  [get_ports {an[1]}]
set_property PACKAGE_PIN R6   [get_ports {an[2]}]
set_property PACKAGE_PIN T6   [get_ports {an[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports {seg[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[*]}]

################################################################################
# Timing Constraints
################################################################################
# False paths for switch and button inputs (asynchronous)
set_false_path -from [get_ports {sw[*]}]
set_false_path -from [get_ports {btn[*]}]
set_false_path -from [get_ports rst_n]

# Output delay for LCD SPI (relax timing for external interface)
set_output_delay -clock [get_clocks sys_clk_pin] -max 5.000 [get_ports lcd_*]
set_output_delay -clock [get_clocks sys_clk_pin] -min 0.000 [get_ports lcd_*]

# LED outputs have relaxed timing
set_output_delay -clock [get_clocks sys_clk_pin] -max 10.000 [get_ports {led[*]}]
set_output_delay -clock [get_clocks sys_clk_pin] -min 0.000 [get_ports {led[*]}]

################################################################################
# Configuration
################################################################################
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

################################################################################
# Additional Constraints for Synthesis
################################################################################
# Allow up to 20% timing slack reduction for better routability
set_property SEVERITY {Warning} [get_drc_checks TIMING-10]

################################################################################
# ALTERNATIVE PIN MAPPINGS FOR OTHER ZYNQ-7000 BOARDS
################################################################################

# ===== PYNQ-Z1 Board =====
# Uncomment the following and comment out ZedBoard mappings above
#
# # Clock (125MHz on PYNQ, adjust constraint)
# #set_property PACKAGE_PIN H16 [get_ports clk]
# #create_clock -period 8.000 -name sys_clk_pin [get_ports clk]
#
# # Switches (directly on PYNQ)
# #set_property PACKAGE_PIN M20 [get_ports {sw[0]}]
# #set_property PACKAGE_PIN M19 [get_ports {sw[1]}]
# # ... add more as needed
#
# # Buttons (directly on PYNQ)
# #set_property PACKAGE_PIN D19 [get_ports {btn[0]}]
# #set_property PACKAGE_PIN D20 [get_ports {btn[1]}]
#
# # LEDs (directly on PYNQ)
# #set_property PACKAGE_PIN R14 [get_ports {led[0]}]
# #set_property PACKAGE_PIN P14 [get_ports {led[1]}]
# # ... and so on

################################################################################
# END OF CONSTRAINTS
################################################################################
