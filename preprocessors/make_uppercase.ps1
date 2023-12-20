dir others -r | % { if ($_.Name -cne $_.Name.ToUpper()) { ren $_.FullName $_.Name.ToUpper() } }
dir stealing -r | % { if ($_.Name -cne $_.Name.ToUpper()) { ren $_.FullName $_.Name.ToUpper() } }

