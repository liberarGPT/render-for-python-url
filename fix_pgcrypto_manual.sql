-- Manual fix for pgcrypto extension issue
-- Run this directly in your Supabase SQL Editor

-- Step 1: Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Step 2: Drop existing functions if they exist
DROP FUNCTION IF EXISTS public.encrypt_api_key(TEXT);
DROP FUNCTION IF EXISTS public.decrypt_api_key(BYTEA);

-- Step 3: Create the encryption function
CREATE OR REPLACE FUNCTION public.encrypt_api_key(key_text TEXT)
RETURNS BYTEA
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  encryption_key TEXT;
BEGIN
  -- Get encryption key from environment
  encryption_key := current_setting('app.encryption_key', true);
  
  -- If no key set, use a default for development
  IF encryption_key IS NULL OR encryption_key = '' THEN
    encryption_key := 'default-development-key-change-in-production';
  END IF;
  
  -- Use pgp_sym_encrypt with the key
  RETURN pgp_sym_encrypt(key_text, encryption_key);
END;
$$;

-- Step 4: Create the decryption function
CREATE OR REPLACE FUNCTION public.decrypt_api_key(encrypted_data BYTEA)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  encryption_key TEXT;
BEGIN
  IF encrypted_data IS NULL THEN
    RETURN NULL;
  END IF;
  
  -- Get encryption key from environment
  encryption_key := current_setting('app.encryption_key', true);
  
  -- If no key set, use a default for development
  IF encryption_key IS NULL OR encryption_key = '' THEN
    encryption_key := 'default-development-key-change-in-production';
  END IF;
  
  -- Use pgp_sym_decrypt with the key
  RETURN pgp_sym_decrypt(encrypted_data, encryption_key);
END;
$$;

-- Step 5: Grant permissions
GRANT EXECUTE ON FUNCTION public.encrypt_api_key(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION public.decrypt_api_key(BYTEA) TO authenticated;

-- Step 6: Test the functions
DO $$
DECLARE
  test_text TEXT := 'test-api-key-123';
  encrypted_data BYTEA;
  decrypted_text TEXT;
BEGIN
  -- Test encryption
  SELECT public.encrypt_api_key(test_text) INTO encrypted_data;
  RAISE NOTICE 'Encryption successful';
  
  -- Test decryption
  SELECT public.decrypt_api_key(encrypted_data) INTO decrypted_text;
  
  IF decrypted_text = test_text THEN
    RAISE NOTICE 'SUCCESS: Encryption and decryption working correctly';
  ELSE
    RAISE NOTICE 'ERROR: Decrypted text does not match original';
  END IF;
END $$;
